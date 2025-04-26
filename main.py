import spacy
import os
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import wikipedia
import requests
import datetime as dt
import re

def contains_math_expression(text):
    cleaned = re.sub(r'[^\d\+\-\*/\(\)\.\s]', '', text)
    return bool(re.search(r'\d+\s*[\+\-\*/]\s*\(?\d+', cleaned))

import nltk
#nltk.download('wordnet')
import requests
requests.Session.proxies = {}
from sklearn.feature_extraction.text import TfidfVectorizer

class ChatbotModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings=None):
            self.model = None
            self.intents_path = intents_path
            self.documents = []
            self.vocabulary = []
            self.intents = []
            self.intents_responses = {}
            self.function_mappings = function_mappings
            self.X = None
            self.y = None
            self.vectorizer = TfidfVectorizer()
            self.last_topic_candidate = None
            self.user_name = None # Variable to store the user's name
            # Load the spaCy model for NER
            self.nlp = spacy.load("en_core_web_md")

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words

    #@staticmethod
    #def bag_of_words(self, words):
    #    return [1 if word in words else 0 for word in self.vocabulary]
    
    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as file:
                intents_data = json.load(file)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.documents.append((' '.join(pattern_words), intent['tag']))  # Join words as a single string for TF-IDF

            # Now we will fit the vectorizer on the documents
            patterns = [doc[0] for doc in self.documents]  # Extract the text patterns
            self.vectorizer.fit(patterns)  # Fit the TF-IDF model
    
    def prepare_data(self):
        bags = []
        indices = []

        for doc in self.documents:
            # Use TF-IDF vectorizer to transform the document into a vector
            bag = self.vectorizer.transform([doc[0]]).toarray()  # Convert to numpy array
            intents_index = self.intents.index(doc[1])  # Get the index for the intent
            bags.append(bag)
            indices.append(intents_index)

        self.X = np.vstack(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        print(f"Shape of input data (self.X): {self.X.shape}")
        X_tensor = torch.tensor(self.X.reshape(len(self.X), -1), dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        input_size = self.X.shape[1]  # Get the number of features in the TF-IDF representation
        self.model = ChatbotModel(input_size, len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss
            
            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")


    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as f:
            json.dump({ 'input_size': self.X.shape[1], 'output_size': len(self.intents) }, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

            self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])    
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            self.parse_intents()
            self.prepare_data()

            # Reinitialize the vectorizer with correct feature dimension
            input_size = self.X.shape[1]
            output_size = self.model.fc3.out_features
            self.model = ChatbotModel(input_size, output_size)
            self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, input_message):
        if contains_math_expression(input_message):
            print("Give me a moment to solve this.")
            return self.function_mappings['math'](input_message)
        
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.vectorizer.transform([' '.join(words)]).toarray()  # Convert user input to a TF-IDF vector

        bag_tensor = torch.tensor(bag, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if predicted_intent == "wiki":
            # Clean the input for better Wikipedia search terms
            cleaned = input_message.lower().replace("who is", "").replace("tell me about", "").replace("what is", "").strip()
            self.last_topic_candidate = cleaned

        # Perform NER on the input message
        doc = self.nlp(input_message)

        # Store user's name if identified and not already stored
        if doc.ents and self.user_name is None:
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    self.user_name = ent.text
                    break # Assume the first PERSON entity is the user's name

        if self.function_mappings and predicted_intent in self.function_mappings:
            # Return the result of the function mapping and the identified entities
            return self.function_mappings[predicted_intent](), doc.ents

        if self.intents_responses[predicted_intent]:
            # Return a random response from intents and the identified entities
            return random.choice(self.intents_responses[predicted_intent]), doc.ents
        else:
            # Return None response and the identified entities
            return None, doc.ents

        # Print the identified entities
        if doc.ents:
            print("Identified Entities:")
            for ent in doc.ents:
                print(f"  - {ent.text} ({ent.label_})")
        # else:
        #     print("No entities identified.") # Optional: uncomment to see when no entities are found

        # Return the response and the identified entities
        return response, doc.ents

def get_stocks():
    tickers = ['AAPL', 'META', 'NVDA', 'MSFT', 'GOOGL']
    stock_data = yf.download(tickers, period="1d", interval="1d")
    closing_prices = stock_data['Close'].iloc[-1]

    response = "📈 Here are the latest stock prices:\n"
    for ticker in tickers:
        price = closing_prices[ticker]
        response += f"{ticker}: ${price:.2f}\n"

    return response

def search_wikipedia():
    return "__wiki__"

def get_weather():
    city = input("🌤️ Enter your city: ")
    Base_URL = "http://api.openweathermap.org/data/2.5/weather?"
    api_key = '16e791bff0ec0642d76d8d4f1ca6ff0c'  # Replace with your actual OpenWeatherMap API key
    #url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    url = Base_URL + "appid=" + api_key + "&q=" + city
    response = requests.get(url).json()

    #print(response)  # Print the response for debugging
    def kelvin_to_celsius(kelvin):
        celcius = kelvin - 273.15
        farenheit = (kelvin - 273.15) * 9/5 + 32
        return celcius, farenheit

    
    if response.get("cod") == 200:  # Check if the response code is OK (200)
        main = response["main"]
        weather = response["weather"][0]
        temperature = response["main"]["temp"]
        celsius, fahrenheit = kelvin_to_celsius(temperature)
        description = weather["description"]
        return f"🌤️ Weather in {city}: {description}, {celsius}°C"
    else:
        return "❌ City not found or invalid API request. Please try again."

    

    


def solve_math(expression=None):
    expression = input("🧮 Enter a math expression: ")
    
    # Extract only numbers/operators using regex
    match = re.findall(r'[\d\.\+\-\*/\(\)\s]+', expression)
    
    if not match:
        return "❌ Couldn't extract a valid math expression. Try again."

    clean_expr = ''.join(match)
    
    try:
        result = eval(clean_expr)
        return f"The result is: {result}"
    except:
        return "❌ Invalid math expression. Try again."
    


if __name__ == '__main__':
    wiki_mode = False
    last_topic = None

    assistant = ChatbotAssistant('intents.json', function_mappings={'stocks': get_stocks, 'wiki': search_wikipedia, 'weather': get_weather, 'math': solve_math})

    #assistant.load_model('chatbot_model.pth', 'dimensions.json')
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=100)

    while True:
        message = input('Enter your message:')

        if message == '/quit':
            break
                # Check for weather-related queries
        if message.lower() in ["what's the weather like?", "weather update"]:
            print(get_weather())  # Call the weather function directly
            continue


        if wiki_mode:
            follow_up_keywords = ['his', 'her', 'age', 'companies', 'net worth', 'more']
            if any(word in message.lower() for word in follow_up_keywords):
                topic = assistant.last_topic_candidate or last_topic
                topic += " " + message.lower()  # Explicitly add the follow-up topic (e.g., "Elon Musk age")
            else:
                topic = message
            print("🔍 Searching Wikipedia...")
            try:
                summary = wikipedia.summary(topic, sentences=2)
                print(f"📚 {summary}")
                last_topic = topic
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"❗ Too many results. Try being more specific. Options: {e.options[:5]}")
            except wikipedia.exceptions.PageError:
                print("❌ I couldn't find any information on that. Try another question.")
            wiki_mode = False
            continue

        response, entities = assistant.process_message(message)

        # --- Using identified entities to improve responses ---
        modified_response = response # Start with the original response

        if entities:
            # Example: If a PERSON entity is found, try to include it in the response
            for ent in entities:
                if ent.label_ == "PERSON":
                    # This is a very basic example. More sophisticated logic would be needed
                    # to integrate the entity naturally into different response types.
                    if response and response != "__wiki__": # Avoid modifying function call responses
                         modified_response = f"Okay, I can tell you about {ent.text}. " + response
                    break # Only use the first PERSON entity found for this example

            # You can add more conditions here to handle other entity types (ORG, GPE, DATE, etc.)
            # and modify the response accordingly.

        # --- End of using identified entities ---


        if modified_response == "__wiki__":
            wiki_mode = True
            print("🧠 Sure! What would you like to learn about?")
        elif modified_response:
            # --- Fill entity placeholders in the response ---
            final_response = modified_response
            if entities:
                for ent in entities:
                    placeholder = f"[{ent.label_}]"
                    # Replace the first occurrence of the specific entity type placeholder
                    if placeholder in final_response:
                        final_response = final_response.replace(placeholder, ent.text, 1)
                    # Also handle a generic [ENTITY] placeholder
                    elif "[ENTITY]" in final_response:
                         final_response = final_response.replace("[ENTITY]", ent.text, 1)
                    # Handle [TOPIC] placeholder (used in language_help example)
                    elif "[TOPIC]" in final_response and ent.label_ in ["LANGUAGE", "SKILL", "TOPIC"]: # Add relevant labels here
                         final_response = final_response.replace("[TOPIC]", ent.text, 1)


            # --- End of filling placeholders ---

            # Replace hardcoded "Anas" with the identified user name if available
            if assistant.user_name:
                final_response = final_response.replace("Anas", assistant.user_name)

            print(final_response)
        else:
            # Handle cases where the original response was None and no entities led to a modification
            print("I'm not sure how to respond to that.")
