import json
from math import e
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

ticket_classification = TicketClassification(
    category=TicketCategory.ORDER_ISSUE,
    urgency=TicketUrgency.HIGH,
    sentiment=CustomerSentiment.ANGRY,
    confidence=0.9,
    key_information=["Order #12345", "Received tablet instead of laptop"],
    suggested_action="Contact customer to arrange laptop delivery"
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
        self.os_primary_model = "llama3.1"
        self.os_alternative_model = "mistral"
        self.paid_model_name = "gpt-4o"

    def classify_ticket(self, ticket_text: str) -> str:
        #return ticket_classification.model_dump_json(indent=2)
            
        client = OllamaInstructorClient(debug=True)
        try:
            print(f"Processing data using model: {self.os_primary_model}\n\n")
            response = client.chat_completion(
                model=self.os_primary_model,
                pydantic_model=TicketClassification,
                format='json',
                messages=[
                    {
                        "role": "assistant",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user", 
                        "content": ticket_text
                    },
                ],
                options={"temperature": 0},
            )
        except Exception as e:
            print(f"\n\nError: {e}\nRetrying with alternative model: {self.os_alternative_model}\n")
            response = client.chat_completion(
                model=self.os_alternative_model,
                pydantic_model=TicketClassification,
                format='json',
                messages=[
                    {
                        "role": "assistant",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user", 
                        "content": ticket_text
                    },
                ],
                options={"temperature": 0},
            )
            
        return TicketClassification.model_validate_json(response["message"]["content"]).model_dump_json(indent=2)

    # def classify_ticket(self, ticket_text: str) -> str:
    #     response = ollama.chat(
    #         model=self.model_name,
    #         messages=[
    #             {
    #                 "role": "system",
    #                 "content": self.system_prompt,
    #             },
    #             {"role": "user", "content": ticket_text},
    #         ],
    #         options={"temperature": 0},
    #         stream=False,
    #         format="json",
    #     )
    #     return TicketClassification.model_validate_json(response["message"]["content"]).model_dump_json(indent=2)

    # def classify_ticket(self, ticket_text: str) -> str:
    #     client = instructor.patch(OpenAI())
    #     response = client.chat.completions.create(
    #         model=self.paid_model_name,
    #         response_model=TicketClassification,
    #         temperature=0,
    #         max_retries=3,
    #         messages=[
    #             {
    #                 "role": "system",
    #                 "content": self.system_prompt,
    #             },
    #             {
    #                 "role": "user",
    #                 "content": ticket_text
    #             }
    #         ]
    #     )
    #     return response.model_dump_json(indent=2)
