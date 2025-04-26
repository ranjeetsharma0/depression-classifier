# app.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizerFast, RobertaModel
import praw
from fastapi import FastAPI

app = FastAPI()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reddit API Credentials
REDDIT_CLIENT_ID = '5akV9jKTyHgnzRVWHPGh9w'
REDDIT_CLIENT_SECRET = '_sca_sUFPiLh85Opv67owifrfa4jgA'
REDDIT_USER_AGENT = '/u/TryHot6147'

# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Define models
class MLPBranch(nn.Module):
    def __init__(self, input_dim, fc_dim, dropout):
        super(MLPBranch, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.fc(x)

class DepressionClassifier(nn.Module):
    def __init__(self, num_numerical_features=5, dropout=0.5, fc_dim=128, roberta_model_name='roberta-base'):
        super(DepressionClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        roberta_hidden_size = self.roberta.config.hidden_size
        self.mlp_branch = MLPBranch(num_numerical_features, fc_dim, dropout)
        combined_dim = roberta_hidden_size + fc_dim
        self.fc1 = nn.Linear(combined_dim, fc_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(fc_dim, 1)
    
    def forward(self, roberta_input_ids, roberta_attention_mask, numerical_features):
        roberta_cls = self.roberta(roberta_input_ids, attention_mask=roberta_attention_mask).last_hidden_state[:, 0, :]
        numerical_out = self.mlp_branch(numerical_features)
        combined = torch.cat([roberta_cls, numerical_out], dim=1)
        x = self.dropout(F.relu(self.fc1(combined)))
        return torch.sigmoid(self.out(x))

# Load model
MODEL_PATH = "./depressionff_classifier.pth"
model = DepressionClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Reddit API Authentication
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Function to classify
def classify_text(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    numerical_features = torch.zeros((1, 5)).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, numerical_features)
        prediction = (outputs > 0.5).float().item()

    return "Depressed" if prediction == 1 else "Not Depressed"

# Define FastAPI endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the Reddit Depression Classifier API."}

@app.get("/classify_subreddit/{subreddit_name}")
def classify_subreddit(subreddit_name: str, limit: int = 10):
    subreddit = reddit.subreddit(subreddit_name)
    results = []

    for post in subreddit.hot(limit=limit):
        text = post.title + " " + post.selftext
        classification = classify_text(text)
        results.append({
            "title": post.title,
            "classification": classification,
            "url": post.url
        })
    return results
