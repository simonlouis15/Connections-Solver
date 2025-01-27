# Connections-Solver
NLP project designed to automatically solve the NYT word game, "Connections", where the objective is to group 16 words into 4 groups based on semantic similarity, context usage and related word structures.

To use simply clone the file and run it. It is currently set to a previous connections word list and is finding the most likely group of 4 words. To try it with todays, enter the words in today's game into the "words_list" array. After trying the combination the solver gives you, you can remove those 4 words from words_list and run it again to get the next most likely grouping.

This project is currently in the development phase and as such may produce errors in groupings. However, it has shown a high average accuracy rate when determining group topics and discerning semantic similarity using the "word2vec-google-news-300" pre-trained model and HuggingFace's sentence transformer. 

Sample Usage:
![image](https://github.com/user-attachments/assets/c486cc64-0035-4ffe-832e-71edb79c09bd)
![image](https://github.com/user-attachments/assets/37168a90-9939-4665-8c72-214bfa4ba012)
