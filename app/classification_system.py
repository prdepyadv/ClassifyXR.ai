import json
import instructor
import ollama
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from enum import Enum
from typing import List, Dict, Any
from openai import OpenAI
import requests
from ollama_instructor.ollama_instructor_client import OllamaInstructorClient


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

    @classmethod
    def from_openai_response(cls, response_text: str) -> "TicketClassification":
        response_dict = eval(response_text)
        print("response_dict", response_dict)
        return cls(
            category=TicketCategory(response_dict["category"]),
            urgency=TicketUrgency(response_dict["urgency"]),
            sentiment=CustomerSentiment(response_dict["sentiment"]),
            confidence=response_dict["confidence"],
            key_information=response_dict["key_information"],
            suggested_action=response_dict["suggested_action"],
        )

    @classmethod
    def from_api_response(cls, response_text: str) -> "TicketClassification":
        response_dict = json.loads(response_text)
        print("response_dict", response_dict)
        return cls(
            category=TicketCategory(response_dict["category"]),
            urgency=TicketUrgency(response_dict["urgency"]),
            sentiment=CustomerSentiment(response_dict["sentiment"]),
            confidence=response_dict["confidence"],
            key_information=response_dict["key_information"],
            suggested_action=response_dict["suggested_action"],
         )

class ClassificationSystem:
    def __init__(self):
        self.system_prompt = """
            You are an AI assistant for a large e-commerce platform's customer support team. Your role is to analyze incoming customer support tickets and provide structured information to help our team respond quickly and effectively.
            
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
            Analyze the following customer support ticket and provide the requested information in the specified format.
        """

    def classify_ticket(self, ticket_text: str) -> TicketClassification:
        client = OllamaInstructorClient(debug=True)
        response = client.chat_completion(
            model="llama3.1",
            pydantic_model=TicketClassification,
            format='',
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {"role": "user", "content": ticket_text},
            ],
            options={"temperature": 0},
        )
        response_content = response["message"]["content"]

        # response_content = {
        #     "category": "order_issue",
        #     "urgency": "high",
        #     "sentiment": "angry",
        #     "confidence": 0.9,
        #     "key_information": [{"order_number": "12345"}],
        #     "suggested_action": "Initiate an immediate investigation and resolution process to correct the order error."
        # }
        print("Raw response content:", response_content)
        
        try:
            classification = TicketClassification.parse_raw(response_content)
            return classification
        except ValidationError as e:
            print("Validation error:", e)
            raise


'''
    def classify_ticket_ollama_api(self, ticket_text: str) -> TicketClassification:
        response = ollama.chat(
            model="llama3.1",
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {"role": "user", "content": ticket_text},
            ],
            options={"temperature": 0},
            stream=False,
            format="json",
        )
        return TicketClassification.from_api_response(response["message"]["content"])

    def classify_ticket(self, ticket_text: str) -> TicketClassification:
        client = instructor.patch(OpenAI())
        response = client.chat.completions.create(
            model="gpt-4o",
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
        return response
'''
