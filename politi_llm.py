# Part 1: Fine-Tuning the Model (Run Once to Create Model)
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import evaluate
import numpy as np

# Load the LIAR dataset
dataset = load_dataset("liar")

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["statement"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up the model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)
# Configure LoRA for efficient fine-tuning
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(model, config)

# Define evaluation metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
)

# Create trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)
trainer.train()

# Save the fine-tuned model
trainer.save_model("politi_llm")
tokenizer.save_pretrained("politi_llm")

# Part 2: Application with Gradio Interface (For End Users)
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Determine base path for bundled files (used when packaged)
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

# Load models
model_dir = os.path.join(base_path, "politi_llm")
classification_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
classification_tokenizer = AutoTokenizer.from_pretrained(model_dir)
summarizer = pipeline("summarization", model="t5-small")

# Classification function
def classify(input_text=None, input_audio=None, input_csv=None):
    statement = input_text
    results = []
    
    # Handle voice input
    if input_audio is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            with open(input_audio, "rb") as audio_file:
                f.write(audio_file.read())
            audio_file = f.name
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        try:
            statement = r.recognize_google(audio)
        except sr.UnknownValueError:
            statement = "Could not understand audio"
        except sr.RequestError as e:
            statement = f"Could not request results; {e}"
        os.remove(audio_file)
    
    # Handle CSV input for multiple statements
    if input_csv is not None:
        df = pd.read_csv(input_csv)
        statements = df['statement'].tolist()
        for stmt in statements:
            inputs = classification_tokenizer(stmt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = classification_model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            labels = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
            results.append(labels[predicted_class])
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        pd.Series(results).value_counts().plot(kind='bar')
        plt.title("Distribution of Classification Labels")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.savefig(os.path.join(base_path, "classification_graph.png"))
        plt.close()
        
        # Save results to CSV
        df['classification'] = results
        df.to_csv(os.path.join(base_path, "classification_results.csv"), index=False)
        
        return ("Results processed", statement, os.path.join(base_path, "classification_graph.png"), os.path.join(base_path, "classification_results.csv"))
    
    # Single statement classification
    if statement:
        inputs = classification_tokenizer(statement, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = classification_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        labels = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
        score = torch.softmax(logits, dim=1)[0, predicted_class].item()
        result = f"Label: {labels[predicted_class]}, Score: {score:.4f}"
        
        # Generate TTS
        tts = gTTS(result)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tts.save(f.name)
            audio_file = f.name
        
        return (result, audio_file, None, None)
    
    return ("No input provided", None, None, None)

# Summarization function
def summarize(input_text=None, input_file=None):
    if input_file is not None:
        with open(input_file, "r") as f:
            text = f.read()
    else:
        text = input_text
    
    if text:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
        
        # Generate TTS
        tts = gTTS(summary)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tts.save(f.name)
            audio_file = f.name
        
        # Save summary to text file
        with open(os.path.join(base_path, "summary.txt"), "w") as f:
            f.write(summary)
        
        return (summary, audio_file, os.path.join(base_path, "summary.txt"))
    
    return ("No input provided", None, None)

# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# politi LLM")
    gr.Markdown("Classify political statements or summarize historical texts.")
    
    with gr.Tab("Classification"):
        with gr.Row():
            text_input = gr.Textbox(lines=5, label="Enter a political statement")
            audio_input = gr.Audio(source="microphone", label="Or record your voice", type="filepath")
            csv_input = gr.File(label="Or upload CSV for multiple statements")
        classify_button = gr.Button("Classify")
        text_output = gr.Textbox(label="Classification Result")
        audio_output = gr.Audio(label="Listen to result", type="filepath")
        graph_output = gr.Image(label="Graph")
        file_output = gr.File(label="Download Results")
        classify_button.click(
            fn=classify,
            inputs=[text_input, audio_input, csv_input],
            outputs=[text_output, audio_output, graph_output, file_output]
        )
    
    with gr.Tab("Summarization"):
        text_input_sum = gr.Textbox(lines=10, label="Text to summarize")
        file_input_sum = gr.File(label="Or upload a file")
        summarize_button = gr.Button("Summarize")
        summary_output = gr.Textbox(label="Summary")
        audio_output_sum = gr.Audio(label="Listen to summary", type="filepath")
        file_output_sum = gr.File(label="Download Summary")
        summarize_button.click(
            fn=summarize,
            inputs=[text_input_sum, file_input_sum],
            outputs=[summary_output, audio_output_sum, file_output_sum]
        )

# Launch (commented out for packaging)
# iface.launch(server_port=7860)
