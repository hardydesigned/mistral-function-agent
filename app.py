import json
import ollama
import asyncio

from ragapp import rag_agent

data = [
    { "transaction_id": 'T1001', "customer_id": 'C001', "payment_amount": 125.50, "payment_date": '2021-10-05', "payment_status": 'Paid' },
    { "transaction_id": 'T1002', "customer_id": 'C002', "payment_amount": 89.99, "payment_date": '2021-10-06', "payment_status": 'Unpaid' },
    { "transaction_id": 'T1003', "customer_id": 'C003', "payment_amount": 120.00, "payment_date": '2021-10-07', "payment_status": 'Paid' },
    { "transaction_id": 'T1004', "customer_id": 'C002', "payment_amount": 54.30, "payment_date": '2021-10-05', "payment_status": 'Paid' },
    { "transaction_id": 'T1005', "customer_id": 'C001', "payment_amount": 210.20, "payment_date": '2021-10-08', "payment_status": 'Pending' }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "getPaymentStatus",
            "description": "Get payment status of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transactionId": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transactionId"],
            },
        },
    },{
        "type": "function",
        "function": {
            "name": "getPaymentDate",
            "description": "Get the payment date of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transactionId": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transactionId"],
            },
        },
    }
]

def get_payment_status(transaction_id):
    transaction = next((row for row in data if row['transaction_id'] == transaction_id), None)
    if transaction:
        return json.dumps({"status": transaction['payment_status']})
    return json.dumps({"error": "transaction id not found."})

def get_payment_date(transaction_id):
    transaction = next((row for row in data if row['transaction_id'] == transaction_id), None)
    if transaction:
        return json.dumps({"date": transaction['payment_date']})
    return json.dumps({"error": "transaction id not found."})

# Available functions for the agent to use
available_functions = {
    "getPaymentDate": get_payment_date,
    "getPaymentStatus": get_payment_status
}

# Define the agent function
def agent(query):
    messages = [
        {"role": "user", "content": query}
    ]

    tool_call = False
    
    for i in range(5):
      
        response = ollama.chat(
            model='llama3.1',
            messages=messages,
            tools=tools
        )
    
        if response['message']['content'] != "" and not tool_call:
            return {"tool_call": tool_call, "response": response['message']['content']}

        try:
            function_object = response['message']['tool_calls'][0]['function']
            function_name = function_object['name']
            function_args = function_object['arguments']["transactionId"]
            function_response = available_functions[function_name](function_args)
            tool_call = True
            messages.append({
                "role": 'tool',
                "name": function_name,
                "content": function_response
        })
        except:
            pass
    
    return {"tool_call": tool_call, "response": messages[-1]['content']}

#query = "December 25th is on a Sunday, do I get any extra time off to account for that?"
query = "When was the transaction T1001 paid?"

response = agent(query)
if response["tool_call"] == False:
    print("Calling RAG agent")
    print(rag_agent(query))

print(response["response"])
