import pandas as pd
from transformers import pipeline   # to use Hugging Face pre-trained models
import torch                        # to use GPU if available

print("GPU available:", torch.cuda.is_available())

# === LOAD THE REQUIREMENTS DATASET ===
df_train = pd.read_csv("../../dataset/train.csv")
df_test = pd.read_csv("../../dataset/test.csv")


# === LOAD PRE-TRAINED CLASSIFICATION MODELS ===

urgency_classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0)
origin_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)


# === LABELING FUNCTIONS ===
def assign_change(Requirement):
    # Keyword-based classification
    keywords = {
        "addition": ["add", "include", "new", "support", "implement"],
        "deletion": ["remove", "delete", "eliminate", "discard"],
        "modification": ["update", "change", "modify", "adjust", "revise"]
    }
    req_lower = Requirement.lower()             # converts the requirement to lowercase
    for label, words in keywords.items():
        if any(word in req_lower for word in words):
            return label
    return "modification"                       # default fallback

def assign_urgency(Requirement):
    try:
        result = urgency_classifier(Requirement)[0]
        stars = int(result['label'][0])
        return "soon" if stars >= 4 else "later"
    except Exception as e:
        print(f"Error in assign_urgency: {e}")
        return None


def assign_origin(Requirement):
    try:
        labels = ["organization", "market", "customer", "developer knowledge", "project vision"]
        prediction = origin_classifier(Requirement, candidate_labels=labels)        # returns a probability score for each label
        return prediction['labels'][0]      # takes the highest score
    except Exception as e:
        print(f"Error in assign_origin: {e}")
        return None


# === APPLYING LABELS TO EACH REQUIREMENT ===
df_train['Change'] = df_train['Requirement'].apply(assign_change)
df_train['Urgency'] = df_train['Requirement'].apply(assign_urgency)
df_train['Origin'] = df_train['Requirement'].apply(assign_origin)

df_test['Change'] = df_test['Requirement'].apply(assign_change)
df_test['Urgency'] = df_test['Requirement'].apply(assign_urgency)
df_test['Origin'] = df_test['Requirement'].apply(assign_origin)


# === SAVE LABELED DATASETS ===
df_train.to_csv("../../dataset/train_labeled.csv", index=False)
print("✅ Labels assigned and saved in train_labeled.csv")

df_test.to_csv("../../dataset/test_labeled.csv", index=False)
print("✅ Labels assigned and saved in test_labeled.csv")