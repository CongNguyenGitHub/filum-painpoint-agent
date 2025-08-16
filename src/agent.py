import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Dict, List, Optional, Any

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class PainPointAgent:
    """
    Pain Point to Solution Agent for Filum.ai

    This agent matches customer pain points with relevant Filum.ai features
    using a hybrid approach combining keyword matching and semantic similarity.
    """

    def __init__(self, knowledge_base_path: str = None):
        """
        Initialize the agent with knowledge base and ML models

        Args:
            knowledge_base_path: Path to JSON file containing feature knowledge base
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Initialize semantic model (using a lighter model for demonstration)
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            print("Warning: Could not load semantic model. Falling back to keyword matching only.")
            self.semantic_model = None

        # Load knowledge base
        if knowledge_base_path:
            self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        else:
            self.knowledge_base = self.create_sample_knowledge_base()

        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )

        # Prepare feature texts for matching
        self.prepare_feature_texts()

    def create_sample_knowledge_base(self) -> List[Dict]:
        """Create a sample knowledge base based on Filum.ai features"""
        return [
            {
                "id": "VoC_Surveys_PostPurchase",
                "name": "Automated Post-Purchase Surveys",
                "category": "Voice of Customer",
                "sub_category": "Surveys",
                "description": "Automatically sends surveys via email or SMS after purchase to collect feedback consistently",
                "keywords": ["post-purchase", "survey", "feedback", "automated", "consistent collection", "email", "SMS"],
                "sample_pain_points": ["struggling to collect customer feedback consistently after a purchase"],
                "docs_link": "https://filum.ai/docs/voc-surveys",
                "supported_channels": ["Email", "SMS", "Zalo"],
                "related_features": ["Insights_Experience_Analysis"]
            },
            {
                "id": "AI_Agent_FAQ",
                "name": "AI Agent for FAQ & First Response",
                "category": "AI Customer Service",
                "sub_category": "AI Inbox",
                "description": "AI-powered agent that handles frequently asked questions and provides instant first responses",
                "keywords": ["AI agent", "FAQ", "first response", "automated", "deflect", "repetitive questions", "support"],
                "sample_pain_points": ["support agents are overwhelmed by high volume of repetitive questions"],
                "docs_link": "https://filum.ai/docs/ai-customer-service",
                "supported_channels": ["Web", "Mobile", "Zalo", "Email"],
                "related_features": ["Tickets_Management"]
            },
            {
                "id": "Journey_Experience_Analysis",
                "name": "Customer Journey Experience Analysis",
                "category": "Insights",
                "sub_category": "Experience",
                "description": "Analyzes feedback and data across customer journeys to identify friction points and touchpoint issues",
                "keywords": ["customer journey", "touchpoints", "friction", "experience analysis", "frustration", "multi-channel"],
                "sample_pain_points": ["no clear idea which customer touchpoints are causing the most frustration"],
                "docs_link": "https://filum.ai/docs/insights-experience",
                "supported_channels": ["Web", "Mobile", "Zalo", "Email", "POS"],
                "related_features": ["VoC_Surveys_PostPurchase", "Topic_Sentiment_Analysis"]
            },
            {
                "id": "Customer_Profile_360",
                "name": "Customer Profile with Interaction History",
                "category": "Customer 360",
                "sub_category": "Customers",
                "description": "Provides a comprehensive single view of customer interaction history across all touchpoints",
                "keywords": ["customer profile", "interaction history", "single view", "comprehensive", "touchpoints", "360"],
                "sample_pain_points": ["difficult to get a single view of customer's interaction history"],
                "docs_link": "https://filum.ai/docs/customer-360",
                "supported_channels": ["Web", "Mobile", "Zalo", "Email", "POS"],
                "related_features": ["AI_Agent_FAQ", "Tickets_Management"]
            },
            {
                "id": "Topic_Sentiment_Analysis",
                "name": "AI-Powered Topic & Sentiment Analysis",
                "category": "VoC",
                "sub_category": "Conversations/Surveys",
                "description": "Automatically processes text feedback to extract key topics and sentiment, reducing manual analysis time",
                "keywords": ["topic analysis", "sentiment analysis", "automated", "text processing", "survey responses", "themes"],
                "sample_pain_points": ["manually analyzing thousands of open-ended survey responses is too time-consuming"],
                "docs_link": "https://filum.ai/docs/voc-analysis",
                "supported_channels": ["Web", "Mobile", "Zalo", "Email"],
                "related_features": ["Journey_Experience_Analysis"]
            },
            {
                "id": "Multi_Channel_Surveys",
                "name": "Multi-Channel Survey Deployment",
                "category": "Voice of Customer",
                "sub_category": "Surveys",
                "description": "Deploy surveys across Web, Mobile, Zalo, SMS, Email, QR codes, and POS systems",
                "keywords": ["multi-channel", "survey deployment", "web", "mobile", "zalo", "SMS", "email", "QR", "POS"],
                "sample_pain_points": ["need to reach customers across different channels for feedback"],
                "docs_link": "https://filum.ai/docs/voc-surveys-multichannel",
                "supported_channels": ["Web", "Mobile", "Zalo", "SMS", "Email", "QR", "POS"],
                "related_features": ["VoC_Surveys_PostPurchase"]
            },
            {
                "id": "Ticket_Management",
                "name": "Comprehensive Ticket Management System",
                "category": "AI Customer Service",
                "sub_category": "Tickets",
                "description": "Complete ticket management system for tracking and resolving customer support issues",
                "keywords": ["ticket management", "support", "tracking", "resolution", "customer service"],
                "sample_pain_points": ["struggling to track and manage customer support requests efficiently"],
                "docs_link": "https://filum.ai/docs/tickets",
                "supported_channels": ["Web", "Mobile", "Zalo", "Email"],
                "related_features": ["AI_Agent_FAQ", "Customer_Profile_360"]
            }
        ]

    def load_knowledge_base(self, file_path: str) -> List[Dict]:
        """Load knowledge base from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Knowledge base file {file_path} not found. Using sample data.")
            return self.create_sample_knowledge_base()

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for matching

        Args:
            text: Raw text to preprocess

        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Lemmatize words and remove stop words
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]

        return ' '.join(words)

    def prepare_feature_texts(self):
        """Prepare feature texts for TF-IDF vectorization"""
        self.feature_texts = []
        for feature in self.knowledge_base:
            # Combine various text fields for matching
            text_parts = [
                feature.get('name', ''),
                feature.get('description', ''),
                ' '.join(feature.get('keywords', [])),
                ' '.join(feature.get('sample_pain_points', []))
            ]
            combined_text = ' '.join(text_parts)
            preprocessed_text = self.preprocess_text(combined_text)
            self.feature_texts.append(preprocessed_text)

        # Fit TF-IDF vectorizer
        if self.feature_texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.feature_texts)

    def keyword_similarity(self, pain_point: str) -> np.ndarray:
        """
        Calculate keyword-based similarity using TF-IDF

        Args:
            pain_point: User's pain point description

        Returns:
            Array of similarity scores for each feature
        """
        preprocessed_pain_point = self.preprocess_text(pain_point)
        pain_point_vector = self.tfidf_vectorizer.transform([preprocessed_pain_point])
        similarities = cosine_similarity(pain_point_vector, self.tfidf_matrix).flatten()
        return similarities

    def semantic_similarity(self, pain_point: str) -> np.ndarray:
        """
        Calculate semantic similarity using sentence embeddings

        Args:
            pain_point: User's pain point description

        Returns:
            Array of similarity scores for each feature
        """
        if not self.semantic_model:
            return np.zeros(len(self.knowledge_base))

        try:
            # Get embedding for pain point
            pain_point_embedding = self.semantic_model.encode([pain_point])

            # Get embeddings for features
            feature_embeddings = self.semantic_model.encode(self.feature_texts)

            # Calculate cosine similarity
            similarities = cosine_similarity(pain_point_embedding, feature_embeddings).flatten()
            return similarities
        except Exception as e:
            print(f"Warning: Semantic similarity calculation failed: {e}")
            return np.zeros(len(self.knowledge_base))

    def calculate_hybrid_score(self, pain_point: str, keyword_weight: float = 0.4, semantic_weight: float = 0.6) -> np.ndarray:
        """
        Calculate hybrid similarity score

        Args:
            pain_point: User's pain point description
            keyword_weight: Weight for keyword similarity
            semantic_weight: Weight for semantic similarity

        Returns:
            Array of hybrid similarity scores
        """
        keyword_scores = self.keyword_similarity(pain_point)
        semantic_scores = self.semantic_similarity(pain_point)

        # Normalize weights
        total_weight = keyword_weight + semantic_weight
        keyword_weight = keyword_weight / total_weight
        semantic_weight = semantic_weight / total_weight

        hybrid_scores = keyword_weight * keyword_scores + semantic_weight * semantic_scores
        return hybrid_scores

    def filter_by_context(self, scores: np.ndarray, context: Dict) -> np.ndarray:
        """
        Apply context-based filtering to scores

        Args:
            scores: Initial similarity scores
            context: Context information for filtering

        Returns:
            Filtered scores
        """
        if not context:
            return scores

        filtered_scores = scores.copy()

        # Filter by channel if specified
        if 'channel' in context:
            channel = context['channel'].lower()
            for i, feature in enumerate(self.knowledge_base):
                supported_channels = [c.lower() for c in feature.get('supported_channels', [])]
                if supported_channels and not any(ch in channel for ch in supported_channels):
                    filtered_scores[i] *= 0.8  # Reduce score but don't eliminate

        # Filter by customer type if specified
        if 'customer_type' in context:
            customer_type = context['customer_type'].lower()
            # This could be expanded with more sophisticated customer type matching

        # Priority boost
        if 'priority' in context and context['priority'].lower() == 'high':
            # Boost scores for high-priority contexts
            filtered_scores *= 1.1

        return filtered_scores

    def process_pain_point(self, input_data: Dict) -> Dict:
        """
        Main method to process pain point and return suggestions

        Args:
            input_data: Dictionary containing pain_point and optional context

        Returns:
            Dictionary with summary and suggestions
        """
        pain_point = input_data.get('pain_point', '')
        context = input_data.get('context', {})

        if not pain_point.strip():
            return {
                "summary": "Please provide a valid pain point description.",
                "suggestions": []
            }

        # Calculate hybrid similarity scores
        scores = self.calculate_hybrid_score(pain_point)

        # Apply context-based filtering
        filtered_scores = self.filter_by_context(scores, context)

        # Set minimum threshold
        min_threshold = 0.1

        # Get top suggestions
        top_k = 3
        top_indices = np.argsort(filtered_scores)[::-1][:top_k]

        suggestions = []
        valid_suggestions = 0

        for idx in top_indices:
            if filtered_scores[idx] >= min_threshold and valid_suggestions < top_k:
                feature = self.knowledge_base[idx]
                suggestion = {
                    "feature": feature['name'],
                    "category": f"{feature['category']} - {feature['sub_category']}",
                    "description": feature['description'],
                    "how_it_helps": self.generate_help_text(feature, pain_point),
                    "relevance_score": round(float(filtered_scores[idx]), 2),
                    "docs_link": feature.get('docs_link', ''),
                    "integration_notes": self.generate_integration_notes(feature)
                }
                suggestions.append(suggestion)
                valid_suggestions += 1

        if not suggestions:
            return {
                "summary": "No highly relevant solutions found. Please provide more specific details about your pain point or context.",
                "suggestions": []
            }

        summary = f"Based on your pain point, here are the top {len(suggestions)} relevant Filum.ai solutions ranked by relevance."

        return {
            "summary": summary,
            "suggestions": suggestions
        }

    def generate_help_text(self, feature: Dict, pain_point: str) -> str:
        """
        Generate contextual help text for a feature

        Args:
            feature: Feature dictionary
            pain_point: Original pain point

        Returns:
            Help text explaining how the feature addresses the pain point
        """
        base_description = feature.get('description', '')

        # Simple contextual adaptation based on keywords
        pain_lower = pain_point.lower()

        if 'feedback' in pain_lower or 'survey' in pain_lower:
            if 'survey' in feature['name'].lower():
                return f"Automates feedback collection to ensure consistent customer insights."

        if 'overwhelm' in pain_lower or 'repetitive' in pain_lower:
            if 'ai agent' in feature['name'].lower():
                return f"Reduces agent workload by handling routine inquiries automatically."

        if 'touchpoint' in pain_lower or 'frustration' in pain_lower:
            if 'journey' in feature['name'].lower():
                return f"Identifies problem areas across customer touchpoints through data analysis."

        if 'single view' in pain_lower or 'history' in pain_lower:
            if '360' in feature['name'].lower() or 'profile' in feature['name'].lower():
                return f"Consolidates all customer interactions into one comprehensive view."

        if 'manual' in pain_lower or 'time-consuming' in pain_lower:
            if 'analysis' in feature['name'].lower():
                return f"Automates text analysis to save time and reveal insights quickly."

        # Default help text
        return base_description[:100] + "..." if len(base_description) > 100 else base_description

    def generate_integration_notes(self, feature: Dict) -> str:
        """
        Generate integration notes for a feature

        Args:
            feature: Feature dictionary

        Returns:
            Integration notes string
        """
        related_features = feature.get('related_features', [])
        if related_features:
            # Find actual feature names for related features
            related_names = []
            for related_id in related_features[:2]:  # Limit to 2 for brevity
                for kb_feature in self.knowledge_base:
                    if kb_feature['id'] == related_id:
                        related_names.append(kb_feature['name'])
                        break

            if related_names:
                return f"Integrates well with {', '.join(related_names)} for enhanced capabilities."

        return f"Part of the {feature['category']} suite for comprehensive customer experience management."
