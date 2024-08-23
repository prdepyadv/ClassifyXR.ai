import json
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from enum import Enum
from typing import List, Dict, Any, Mapping, Literal
from ollama_instructor.ollama_instructor_client import OllamaInstructorClient

"""
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
"""

"""
Todo:
## Fraud detection in tickets
1. We can add concept to check if ticket is fraud or not based on the other user PII data like IP, Country, etc.
"""


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


class TicketImpact(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TicketSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


ticket_classification_example = {
    "category": TicketCategory.ORDER_ISSUE,
    "sentiment": CustomerSentiment.FRUSTRATED,
    "impact": TicketImpact.MEDIUM,
    "severity": TicketSeverity.HIGH,
    "priority": TicketPriority.HIGH,
    "confidence": 0.95,
    "key_information": ["order number: 12345", "placed a week ago"],
    "specific_requests": ["clarity on order status", "expected delivery date"],
    "suggested_action": "Investigate the issue with the tracking page for order #12345 and provide an update to the customer regarding their order status and expected delivery date.",
}


class TicketClassification(BaseModel):
    category: TicketCategory = Field(
        description="Category of the customer support ticket"
    )
    sentiment: CustomerSentiment = Field(
        description="Sentiment of the customer expressing the issue"
    )
    impact: TicketImpact = Field(
        default=TicketImpact.LOW,
        description="""
            Impact of the ticket on the business operations.
            For example:
            - **high**: If it affects a subset of service outages affecting all users, major data breaches, critical application failures, 
            - **medium**: If it affects a subset of performance issues affecting a specific department, partial service outages, functionality issues with significant but non-critical applications,
            - **low**: If it affects minor issues that do not significantly impact the business operations
        """,
    )
    severity: TicketSeverity = Field(
        default=TicketSeverity.LOW,
        description="""
            Severity of the ticket based on the potential impact on the customer or business
            For example:
            - **critical**: If it involves complete service outage, critical security vulnerabilities, data loss affecting essential operations,
            - **high**: If it involves major functionality issues, significant performance degradation, issues affecting a large number of users,
            - **medium**: If it involves a significant issue that affects the customer or business operations,
            - **low**: If it involves minor issues that do not significantly impact the customer or business operations
        """,
    )
    priority: TicketPriority = Field(
        description="""
            Priority level for handling the ticket based on customer sentiment, impact and severity.
            For example:
            - **high**: Issues that cause a complete service outage or a severe degradation impacting critical business operations. There is no workaround, and the problem requires immediate attention,
            - **medium**: Issues that significantly impact functionality or performance but have a workaround available. These are important problems that need to be resolved quickly to avoid further escalation,
            - **low**: Issues that cause minor impact or inconvenience with no immediate threat to business operations. These can be handled during normal support hours and do not require urgent attention
        """
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

    """
    Ticket: Hі, I'm getting a bit frustrated as I haven't received any updates on my order #12345, placed a week ago.
    Checked the tracking page, but it's still blank. Can you let me know what's going on? A little clarity would be great. When can I expect my order?)
    """
    model_config = ConfigDict(
        json_schema_extra={"examples": [ticket_classification_example]}
    )


class TicketResponse(BaseModel):
    suggested_response: str = Field(
        description="Suggested response to the customer based on the ticket classification"
    )
    confidence: float = Field(
        ge=0, le=1, description="Confidence score for the response"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "suggested_response": """
                        Hi Jake, 
                        
                        Our sincere apologies for the delay we're experiencing to process your order #111296 due to unexpected surge in the demand.On a positive note, your order will be delivered by the end of today.
                        Genuinely appreciate your patience throughout!
                        
                        Thank you for understanding!
                    """,
                    "confidence": 0.9,
                }
            ]
        }
    )


class ClassificationSystem:
    def __init__(self):
        self.assistant_name = "Alex"
        self.assistant_title = "Customer Support"
        self.business_context = f"""
            You are an AI assistant with name {self.assistant_name} and your title is {self.assistant_title} for a large e-commerce platform's customer support team.
            Your role is to analyze incoming customer support tickets and provide structured information to help our team respond quickly and effectively.
            
            Business Context:
            - We handle thousands of tickets daily across various categories (orders, accounts, products, technical issues, sales, feedback, billing).
            - Quick and accurate classification and response is crucial for customer satisfaction and operational efficiency.
        """
        self.os_models = ["llama3.1", "mistral", "gemma:7b-instruct"]
        self.os_primary_model = "llama3.1"
        

    def process_ticket(
        self,
        model_name: str,
        prompt: str,
        pydantic_model,
        user_request: str | None = None,
        format: Literal["", "json"] = "json",
        debug: bool = False,
    ) -> Mapping[str, Any]:
        if model_name not in self.os_models:
            raise ValueError(f"Invalid model name: {model_name}")

        messages = [{"role": "assistant", "content": prompt}]
        if user_request:
            messages.append({"role": "user", "content": user_request})

        print(f"Processing data using model: {model_name}")
        os_client = OllamaInstructorClient(debug=debug)
        return os_client.chat_completion(
            model=model_name,
            pydantic_model=pydantic_model,
            format=format,
            messages=messages,
            options={"temperature": 0.25},
        )



    def classify(self, ticket_text: str) -> dict:
        best_response = None
        highest_confidence = 0.0
        model_used = None
        ticket_classification_prompt = f"""
            {self.business_context}
            
            Your tasks:
            1. Categorize the ticket into the most appropriate category.
            2. Determine the customer's sentiment.
            3. Evaluate the impact on the business (low, medium, high).
            4. Specify the severity of the issue (low, medium, high, critical).
            5. Assign a priority level for handling the ticket (low, medium, high) based on the result of the combination of customer sentiment, impact and severity.
            6. Extract key information that would be helpful for our support team, including specific customer requests (e.g., requested information like delivery dates, order status updates).
            7. Identify and emphasize any specific requests made by the customer (e.g., asking for a delivery date, requesting a refund) and list them separately under 'specific_requests'.
            8. Suggest an initial action for handling the ticket, making sure to address any specific requests made by the customer.
            9. Provide a confidence score for your classification.
            
            Remember:
            - Be objective and base your analysis solely on the information provided in the ticket.
            - If you're unsure about any aspect, reflect that in your confidence score.
            - For 'key_information', extract specific details like order numbers, product names, or account issues.
            - The 'suggested_action' should be a brief, actionable step for our support team.
            - We prioritize ticket based on customer sentiment, impact and severity.

            Analyze the following customer support ticket and provide the requested information in the specified format while adhering to the following JSON schema: {TicketClassification.model_json_schema()}.
        """

        for model in self.os_models:
            try:
                response = self.process_ticket(
                    user_request=ticket_text,
                    prompt=ticket_classification_prompt,
                    model_name=model,
                    pydantic_model=TicketClassification,
                )
                classification = TicketClassification.model_validate(
                    response["message"]["content"]
                )
                if (
                    classification.confidence
                    and classification.confidence > highest_confidence
                ):
                    highest_confidence = classification.confidence
                    best_response = response["message"]["content"]
                    model_used = model

                    # For the sake of performance, if the confidence is already high, we can break the loop
                    if classification.confidence > 0.9:
                        break

            except Exception as e:
                print(
                    f"\n\nError while ticket classification with model {model}: {e}\nContinuing with the next model.\n"
                )

        if not best_response:
            raise ValueError("All models failed to classify the ticket.")

        return {"classification_model": model_used, "classification": best_response}



    def classify_and_response(self, ticket_text: str) -> dict:
        ticket_classification = self.classify(ticket_text)

        # Customer insights can be fetched from the customer database or other sources like CRM
        customer_insights = {
            "name": "Jane Doe",
            "email": "jane.doe@example.com",
            "phone": "+1-555-123-4567",
            "demography": {
                "age": 34,
                "gender": "female",
                "income_level": "middle-income",
            },
            "location": "New York, USA",
            "preferences": {
                "preferred_contact_method": "email",
                "preferred_language": "English",
                "purchase_history": ["electronics", "home appliances", "fashion"],
            },
        }

        # Previous communication history can be fetched from the database or CRM
        previous_communication_history = []
        ticket_response_prompt = f"""
            {self.business_context}

            Contextual Information:
            - You already have classified the following customer support ticket:
                Ticket Text: {ticket_text}
                Ticket Details: {json.dumps(ticket_classification['classification'])}
            - Customer insights such as demography, location, and preferences if any:
                Customer Insights: {json.dumps(customer_insights)}
            - Use previous communication history stored to tailor the response if any:
                Previous Communication History: {json.dumps(previous_communication_history)}
            
            Instructions:
            - Please generate a polite, empathetic, and helpful response to the customer, addressing their concerns.
            - **Do not provide specific information about actions taken (e.g., shipping, investigating) unless explicitly stated.**
            - If you need to mention next steps or an investigation, do so cautiously without assuming actions not confirmed in the input data.
            - Offer a clear resolution or next steps to the customer based on the information provided.
            - Provide a confidence score for your response.
            - Please end the response with a polite sign-off, such as "Best regards" or "Thank you for your patience" with your name and your title.

            Based on this information, please generate a polite and helpful response to the customer, addressing their concerns and providing the information they have requested, while adhering to the following JSON schema: {TicketResponse.model_json_schema()}.
        """
        best_response = None
        highest_confidence = 0.0
        model_used = None

        for model in self.os_models:
            try:
                response = self.process_ticket(
                    user_request=ticket_text,
                    prompt=ticket_response_prompt,
                    model_name=model,
                    pydantic_model=TicketResponse,
                )
                customer_response = TicketResponse.model_validate(
                    response["message"]["content"]
                )
                if (
                    customer_response.confidence
                    and customer_response.confidence > highest_confidence
                ):
                    highest_confidence = customer_response.confidence
                    best_response = response["message"]["content"]
                    model_used = model

                    # For the sake of performance, if the confidence is already high, we can break the loop
                    if customer_response.confidence > 0.9:
                        break

            except Exception as e:
                print(
                    f"\n\nError while ticket response with model {model}: {e}\nContinuing with the next model.\n"
                )

        if not best_response:
            raise ValueError("All models failed to generate a response.")

        ticket_classification["response"] = best_response["suggested_response"]
        ticket_classification["response_model"] = model_used

        return ticket_classification
