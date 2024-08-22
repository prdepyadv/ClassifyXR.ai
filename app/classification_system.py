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
Email Automation feature is a game-changer for businesses looking to streamline their support operations on email.
Leveraging LLMs to identify multiple intents, generating responses and even escalate to the right teams is a technological breakthrough.
Automate upto 80% of incoming email queries to drive quick, empathetic resolutions while reducing operational costs.
Reduce support operations op-ex costs by upto 60%
Reduce in ticket volume by upto 85%
Improvement in First call response FCR by upto 20% 

## Advanced email understanding
#  Powered by LLMs, Yellow.ai Email Automation comprehends complex mail written in everyday language
1. Create email classification system that understands long, unstructured mails
2. Identifies multiple intents in the email & ticket (intent and action items), recognizes urgency and user sentiment.

## Automated workflow triggering
#  Execute relevant downstream workflows automatically based on email intent identified
3. Refer knowledge base for relevant resolutions
4. Escalate to the right team
5. Transfer to agent for complex use cases

## Empathetic, personalized resolutions
#  Go beyond templatized resolutions and give your customers a truly human self-serve experience
6. Cross-refer user details with CRM for contextual resolutions.
7. Identify user sentiment for customized responses.
8. Ensure adherence to company content guidelines while replying.

## AI-powered agent assist
#  Seamless transfer to human agent depending on customer query
9. AI-powered response suggestions to human agent
10. Automated trigger for agent transfer when negative user sentiment detected
11. Human agent looped in as a fallback for specific complex queries

## Connect with existing systems
#  Provide resolutions based on customer insights like demography, location, preferences, and more
12. Seamless integration with underlying systems, including existing ticketing systems and CRM tools
13. Conversations over mail between customers and the email bot recorded in the CRM for reference
14. Access the previous mail history stored in CRM to deliver contextual responses based on past interactions
'''

'''
Todo:
## Fraud detection in tickets
1. We can add concept to check if ticket is fraud or not based on the other user PII data like IP, Country, etc.

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
    specific_requests: List[str] = Field(
        description="List of specific requests made by the customer, such as asking for a delivery date or requesting a refund"
    )
    suggested_action: str = Field(
        description="Brief suggestion for handling the ticket, ensuring it addresses any specific requests made by the customer"
    )

    '''
    Ticket: Hі, I'm getting a bit frustrated as I haven't received any updates on my order #12345, placed a week ago.
    Checked the tracking page, but it's still blank. Can you let me know what's going on? A little clarity would be great. When can I expect my order?)
    '''
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "category": TicketCategory.ORDER_ISSUE,
                    "sentiment": CustomerSentiment.FRUSTRATED,
                    "urgency": TicketUrgency.HIGH,
                    "impact": TicketImpact.MEDIUM,
                    "priority": TicketPriority.HIGH,
                    "confidence": 0.95,
                    "key_information": ["order number: 12345", "placed a week ago"],
                    "specific_requests": ["clarity on order status", "expected delivery date"],
                    "suggested_action": "Investigate the issue with the tracking page for order #12345 and provide an update to the customer regarding their order status and expected delivery date.",
                }
            ]
        }
    )

class TicketResponse(BaseModel):
    suggested_response: str = Field(
        description="Suggested response to the customer based on the ticket classification"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "suggested_response": 
                        """
                        Hi Jake, 
                        
                        Our sincere apologies for the delay we're experiencing to process your order #111296 due to unexpected surge in the demand. On a positive note, your order will be delivered by the end of today.
                        Genuinely appreciate your patience throughout!
                        
                        Thank you for understanding!
                        """,
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
            6. Extract key information that would be helpful for our support team, including specific customer requests (e.g., requested information like delivery dates, order status updates).
            7. Identify and emphasize any specific requests made by the customer (e.g., asking for a delivery date, requesting a refund) and list them separately under 'specific_requests'.
            8. Suggest an initial action for handling the ticket, making sure to address any specific requests made by the customer.
            9. Provide a confidence score for your classification.
            
            Remember:
            - Be objective and base your analysis solely on the information provided in the ticket.
            - If you're unsure about any aspect, reflect that in your confidence score.
            - For 'key_information', extract specific details like order numbers, product names, or account issues.
            - The 'suggested_action' should be a brief, actionable step for our support team.
            Analyze the following customer support ticket and provide the requested information in the specified format while adhering to the following JSON schema: {TicketClassification.model_json_schema()}.
        """
        self.os_models = ["llama3.1", "mistral", "gemma:7b-instruct"]
        self.os_primary_model = "llama3.1"
        self.paid_model_name = "gpt-4o"

    def process_ticket(self, ticket_text: str, model_name: str) -> Mapping[str, Any]:
        if model_name not in self.os_models:
            raise ValueError(f"Invalid model name: {model_name}")

        print(f"Processing data using model: {model_name}")
        os_client = OllamaInstructorClient()
        return os_client.chat_completion(
            model=model_name,
            pydantic_model=TicketClassification,
            format="json",
            messages=[
                {
                    "role": "assistant",
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

        for model in self.os_models:
            try:
                response = self.process_ticket(ticket_text, model)
                classification = TicketClassification.model_validate(response["message"]["content"])
                if classification.confidence and classification.confidence > highest_confidence:
                    highest_confidence = classification.confidence
                    best_response = response["message"]["content"]
                    model_used = model

                    # For the sake of performance, if the confidence is already high, we can break the loop
                    if classification.confidence > 0.9:
                        break
                
            except Exception as e:
                print(f"\n\nError with model {model}: {e}\nContinuing with the next model.\n")
        
        if not best_response:
            raise ValueError("All models failed to classify the ticket.")
            
        return { 
            "model": model_used,
            "classification": best_response
        }
    

    def classify_and_response(self, ticket_text: str) -> dict:
        ticket_classification = self.classify(ticket_text)
        print(f"Classification: {ticket_classification}")

        response_prompt = f"""
        You have classified the following customer support ticket:
        Ticket Text: {ticket_text}
        Ticket Details: {json.dumps(ticket_classification['classification'])}
        
        Based on this information, please generate a polite and helpful response to the customer, addressing their concerns and providing the information they have requested.
        """
    
        print(f"Response Prompt: {response_prompt}")
        os_client = OllamaInstructorClient()
        customer_response = os_client.chat_completion(
            model=self.os_primary_model,
            pydantic_model=TicketResponse,
            format="",
            messages=[
                {"role": "assistant", "content": response_prompt}
            ],
            options={"temperature": 0.5},
        )
        print(f"Customer Response: {customer_response}")
        ticket_classification["suggested_response"] = customer_response["message"]["content"]

        return ticket_classification
            
    
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
