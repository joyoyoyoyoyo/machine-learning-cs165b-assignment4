knn will default to write to file

mahadist.py and kerpercep will need to output via commandline with the '>' character

```
cd hw4-1
python2.7 mahadist.py training_data_prob1_1.txt testing_data_prob1_1.txt > output1.txt
python2.7 mahadist.py training_data_prob1_2.txt testing_data_prob1_2.txt > output2.txt
cd ..

cd hw4-2
python2.7 kerpercep.py 1.0 positive_training_data_prob2_1.txt negative_training_data_prob2_1.txt positive_testing_data_prob2_1.txt negative_testing_data_prob2_1.txt > output1.txt
python2.7 kerpercep.py 1.0 positive_training_data_prob2_2.txt negative_training_data_prob2_2.txt positive_testing_data_prob2_2.txt negative_testing_data_prob2_2.txt > output2.txt
cd ..

cd hw4-3
python2.7 knn.py 1 training_data_prob3_1.txt testing_data_prob3_1.txt > output1.txt
python2.7 knn.py 5 training_data_prob3_2.txt testing_data_prob3_2.txt > output2.txt
cd ..
# Fin

```