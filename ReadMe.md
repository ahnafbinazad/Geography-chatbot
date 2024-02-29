# Geography Chatbot 

This chatbot is designed to assist users with various geography-related queries and tasks. It utilises a combination of APIs, natural language processing techniques, and machine learning models to provide accurate and informative responses. 

## Functionalities

- **Answer Geography Queries**: The chatbot can respond to user queries about geographical locations, countries, continents, and other trivia related to geography.
- **Differentiated Responses**: It employs different switch cases based on the type of questions asked by the user.
- **Wikipedia Integration**: When users ask 'what' or 'who' questions, the chatbot uses the Wikipedia API to provide relevant answers.
- **Weather Information**: For weather-related queries, the chatbot leverages a weather API to furnish the user with accurate weather information.
- **Cosine Similarity Matching**: If a user input's a question pattern not recognised by the chatbot, it utilises cosine similarity to find similar matches within its knowledge base.
- **Logical Knowledge Base**: The chatbot contains a logical knowledge base inferred using the ResolutionProver from NLTK to verify the correctness of user queries and statements about places being countries or continents.
- **Inferencing**: It can infer from user statements and dynamically update its knowledge base, ensuring consistency and accuracy.
- **Game Interaction**: Users can engage the chatbot in games that involve fuzzy logic-based questions and store their answers for evaluation.
- **Flag Recognition**: The chatbot can recognise countries from flag images and videos provided by users, employing a trained machine learning model for accurate predictions.

## Technologies Used

- Wikipedia API
- Weather API
- NLTK (Natural Language Toolkit)
- Googlesearch API
- Text-to-Speech (MacOS system's function)
- ResolutionProver (from NLTK)
- Machine Learning Models (for flag recognition)
- Video Processing Techniques (for flag recognition from videos)
- Convolutional Neural Network (CNN) Model
- Hyperparameter Tuning (Bayesian Optimisation, Random Search, Grid Search)
---


## How to Use

1. **Download Dependencies**:
   - Download the required dependencies using the `requirements.txt` file.

2. **Run the Chatbot**:
   - With the files available, run the `chatbot.py` file and the chatbot will start.

3. **Training Your Own Model**:
   - If you want to train your own model:
     - Use the `image_preprocessing.py` file to process and save training datasets as numpy (npy) files.
     - Use the `cnn_trainer.py` to train and save the training model. Alternatively, use `hyper_para_bayesian_optimisation.py` or `hyper_para_random_search.py` to train with hyperparameter-tuned layers for the CNN model.
   
4. **Run the Chatbot**:
   - After training (if applicable), run the `chatbot.py` file, and you are good to go.

