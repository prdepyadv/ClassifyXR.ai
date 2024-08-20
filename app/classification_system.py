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
1. Add concept to choose the best response based on the confidence score returned by the different models.
2. We can add concept to check if ticket is fraud or not based on the other user PII data like IP, Country, etc.
'''

class TicketCategory(str, Enum):
    ORDER_ISSUE = "order_issue"
    ACCOUNT_ACCESS = "account_access"
    PRODUCT_INQUIRY = "product_inquiry"
    TECHNICAL_SUPPORT = "technical_support"
    BILLING = "billing"
    OTHER = "other"


class CustomerSentiment(str, Enum):
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"


class TicketUrgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketClassification(BaseModel):
    category: TicketCategory
    urgency: TicketUrgency
    sentiment: CustomerSentiment
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
                    "category": "account_access",
                    "urgency": "high",
                    "sentiment": "frustrated",
                    "confidence": 0.9,
                    "key_information": ["pending orders", "password reset issue"],
                    "suggested_action": "Escalate to a senior support agent for further assistance with account access and pending order resolution.",
                }
            ]
        }
    )


ticket_classification = TicketClassification(
    category=TicketCategory.ORDER_ISSUE,
    urgency=TicketUrgency.HIGH,
    sentiment=CustomerSentiment.ANGRY,
    confidence=0.9,
    key_information=["Order #12345", "Received tablet instead of laptop"],
    suggested_action="Contact customer to arrange laptop delivery",
)


class ClassificationSystem:
    def __init__(self):
        self.system_prompt = f"""
            You are an AI assistant for a large e-commerce platform's customer support team.
            Your role is to analyze incoming customer support tickets and provide structured information to help our team respond quickly and effectively
            
            Business Context:
            - We handle thousands of tickets daily across various categories (orders, accounts, products, technical issues, billing).
            - Quick and accurate classification is crucial for customer satisfaction and operational efficiency.
            - We prioritize based on urgency and customer sentiment.
            
            Your tasks:
            1. Categorize the ticket into the most appropriate category.
            2. Assess the urgency of the issue (low, medium, high, critical).
            3. Determine the customer's sentiment.
            4. Extract key information that would be helpful for our support team.
            5. Suggest an initial action for handling the ticket.
            6. Provide a confidence score for your classification.
            
            Remember:
            - Be objective and base your analysis solely on the information provided in the ticket.
            - If you're unsure about any aspect, reflect that in your confidence score.
            - For 'key_information', extract specific details like order numbers, product names, or account issues.
            - The 'suggested_action' should be a brief, actionable step for our support team.
            Analyze the following customer support ticket and provide the requested information in the specified format while adhering to the following JSON schema: {TicketClassification.model_json_schema()}.
        """
        self.os_primary_model = "llama3.1"
        self.os_alternative_model = "mistral"
        self.os_alternative_model_2 = "gemma:7b-instruct"
        self.paid_model_name = "gpt-4o"
        self.os_client = OllamaInstructorClient()

    def process_ticket(self, ticket_text: str, model_name: str) -> Mapping[str, Any]:
        if model_name not in [self.os_primary_model, self.os_alternative_model, self.os_alternative_model_2]:
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
        models = [
            self.os_primary_model,
            self.os_alternative_model,
            self.os_alternative_model_2,
        ]
        best_response = None
        highest_confidence = 0.0
        model_used = None

        for model in models:
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
