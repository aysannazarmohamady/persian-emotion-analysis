import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from utils.data_loader import download_dataset
from utils.preprocessor import preprocess_persian_text
from models.emotion_classifier import build_emotion_model, predict_emotion

def main():
    """
    Main function to run the Persian text emotion analysis
    """
    print("Persian Text Emotion Analysis")
    print("============================")
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Download dataset
    print("\nDownloading dataset...")
    data = download_dataset()
    
    # Display dataset info
    print("\nDataset Information:")
    print(data.head())
    print("\nEmotion Distribution:")
    emotion_counts = data['emotion'].value_counts()
    print(emotion_counts)
    
    # Save distribution chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
    plt.title('Distribution of Emotions in Dataset')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/emotion_distribution.png')
    print("Emotion distribution chart saved to 'results/emotion_distribution.png'")
    
    # Build and evaluate model
    print("\nBuilding and evaluating the model...")
    model, vectorizer, X_test, y_test = build_emotion_model(data)
    
    # Save confusion matrix
    y_pred = model.predict(vectorizer.transform(X_test))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    print("Confusion matrix saved to 'results/confusion_matrix.png'")
    
    # Save classification report
    report = classification_report(y_test, y_pred)
    with open('results/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("Classification report saved to 'results/classification_report.txt'")
    
    # Test model with examples
    test_texts = [
        "خیلی خوشحالم که این پروژه رو تموم کردم",  # Happy
        "از نتیجه آزمون ناراحت و عصبانی هستم",      # Angry/Sad
        "نگرانم که به موقع به قرار ملاقات نرسم",     # Fear
        "امروز هوا خیلی خوب و دلپذیر است"           # Other
    ]
    
    print("\nPrediction Examples:")
    for text in test_texts:
        result = predict_emotion(text)
        print(f"\nText: {result['text']}")
        print(f"Predicted Emotion: {result['predicted_emotion']}")
        print("Probabilities for each class:")
        for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {prob:.4f}")

if __name__ == "__main__":
    main()
