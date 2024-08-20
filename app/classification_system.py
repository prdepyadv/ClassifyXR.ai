import json
from math import e
from multiprocessing import process
import instructor
import ollama
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from enum import Enum
from typing import List, Dict, Any, Mapping
from openai import OpenAI
import requests
from ollama_instructor.ollama_instructor_client import OllamaInstructorClient

'''
Todo: 
1. (done) Add concept to choose the best response based on the confidence score returned by the different models.
2. We can add concept to check if ticket is fraud or not based on the other user PII data like IP, Country, etc.
'''

class TicketCategory(str, Enum):
    ORDER_ISSUE = "order_issue"
    ACCOUNT_ACCESS = "account_access"
    PRODUCT_INQUIRY = "product_inquiry"
    SALES_INQUIRY = "sales_inquiry"
    TECHNICAL_SUPPORT = "technical_support"
    INFRASTRUCTURE = "infrastructure"
    GENERAL_SUPPORT = "general_support"
    BILLING = "billing"
    SHIPPING = "shipping"
    FEEDBACK_SUGGESTIONS = "feedback_suggestions"
    RETURN_REFUND = "return_refund"
    FEATURE_REQUEST = "feature_request"
    OTHER = "other"


class CustomerSentiment(str, Enum):
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    HAPPY = "happy"
    CONFUSED = "confused"


class TicketUrgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    URGENT = "urgent"


class TicketImpact(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    IMMEDIATE = "immediate"

class TicketClassification(BaseModel):
    category: TicketCategory = Field(
        description="Category of the customer support ticket"
    )
    sentiment: CustomerSentiment = Field(    
        description="Sentiment of the customer expressing the issue"
    )
    urgency: TicketUrgency = Field(
        description="Urgency level for handling the ticket based on the customer's needs"
    )
    impact: TicketImpact = Field(
        description="Impact of the ticket on the business operations"
    )
    priority: TicketPriority = Field(
        description="Priority level for handling the ticket based on urgency and impact"
    )
    confidence: float = Field(
        ge=0, le=1, description="Confidence score for the classification"
    )
    key_information: List[str] = Field(
        description="List of key points extracted from the ticket"
    )
    suggested_action: str = Field(
        description="Brief suggestion for handling the ticket"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "category": TicketCategory.ACCOUNT_ACCESS,
                    "sentiment": CustomerSentiment.FRUSTRATED,
                    "urgency": TicketUrgency.HIGH,
                    "impact": TicketImpact.CRITICAL,
                    "priority": TicketPriority.CRITICAL,
                    "confidence": 0.9,
                    "key_information": ["pending orders", "password reset issue"],
                    "suggested_action": "Escalate to a senior support agent for further assistance with account access and pending order resolution.",
                }
            ]
        }
    )


class ClassificationSystem:
    def __init__(self):
        self.system_prompt = f"""
            You are an AI assistant for a large e-commerce platform's customer support team.
            Your role is to analyze incoming customer support tickets and provide structured information to help our team respond quickly and effectively
            
            Business Context:
            - We handle thousands of tickets daily across various categories (orders, accounts, products, technical issues, sales, feedback,  billing).
            - Quick and accurate classification is crucial for customer satisfaction and operational efficiency.
            - We prioritize based on urgency and customer sentiment.
            
            Your tasks:
            1. Categorize the ticket into the most appropriate category.
            2. Determine the customer's sentiment.
            3. Assess the urgency of the issue (low, medium, high, critical, urgent).
            4. Evaluate the impact on the business (low, medium, high, critical).
            5. Assign a priority level for handling the ticket (low, medium, high, critical, immediate) based on the result of the combination of Impact and Urgency.
            6. Extract key information that would be helpful for our support team.
            7. Suggest an initial action for handling the ticket.
            8. Provide a confidence score for your classification.
            
            Remember:
            - Be objective and base your analysis solely on the information provided in the ticket.
            - If you're unsure about any aspect, reflect that in your confidence score.
            - For 'key_information', extract specific details like order numbers, product names, or account issues.
            - The 'suggested_action' should be a brief, actionable step for our support team.
            Analyze the following customer support ticket and provide the requested information in the specified format while adhering to the following JSON schema: {TicketClassification.model_json_schema()}.
        """
        self.os_primary_models = ["llama3.1", "mistral", "gemma:7b-instruct"]
        self.paid_model_name = "gpt-4o"
        self.os_client = OllamaInstructorClient()

    def process_ticket(self, ticket_text: str, model_name: str) -> Mapping[str, Any]:
        if model_name not in self.os_primary_models:
            raise ValueError(f"Invalid model name: {model_name}")

        print(f"Processing data using model: {model_name}")
        return self.os_client.chat_completion(
            model=model_name,
            pydantic_model=TicketClassification,
            format="json",
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {"role": "user", "content": ticket_text},
            ],
            options={"temperature": 0.25},
        )
    
    def classify(self, ticket_text: str) -> dict:
        best_response = None
        highest_confidence = 0.0
        model_used = None

        for model in self.os_primary_models:
            try:
                response = self.process_ticket(ticket_text, model)
                
                classification = TicketClassification.model_validate(response["message"]["content"])
                if classification.confidence and classification.confidence > highest_confidence:
                    highest_confidence = classification.confidence
                    best_response = response["message"]["content"]
                    model_used = model
                
            except Exception as e:
                print(f"\n\nError with model {model}: {e}\nContinuing with the next model.\n")
        
        if not best_response:
            raise ValueError("All models failed to classify the ticket.")
            
        return { 
            "model": model_used,
            "ticket": best_response
        }
            
    
    """
    def classify_using_openai(self, ticket_text: str) -> str:
        client = instructor.patch(OpenAI())
        response = client.chat.completions.create(
            model=self.paid_model_name,
            response_model=TicketClassification,
            temperature=0,
            max_retries=3,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": ticket_text
                }
            ]
        )
        return response.model_dump_json(indent=2)
    """
