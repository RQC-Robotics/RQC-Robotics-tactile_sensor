In this project we try to simulate tactile sensor based on the gerd of optical microfibers. Now, it is a vary fierst steps.
In our sensor, physical number of the output chenal is ~sqrt(n), where n is number of pixels in clasic sensor. And it is grate advantage for using in robotics. But ofcourse, redusing numbers of chenal have a side effect, that you can't decoding any input signal univocal. The goal of this simulation shows that for physical input (prasuar profile, that is possibel in robotics task), we can decoding it prity well even for vary big sensor size (and numbers of pixel as well). As physical profile we used superposition of several gauss (15 and 30). And try to predict this profiel by the outputs of our sensor, using a nerual network.



install

The esiast way to run our cod, is using Google Colab. In this way you need to put folder whith data on your Google Drive.
You can download data heare:  https://drive.google.com/drive/folders/1qfujkRPA81V8XF0RwKlAvepwT75hBL8Y?usp=sharing
Then you shoud open exampeles in Colab. 
main.ipynb is exampel in which:
1) generated presure profieles (you can scip this step because it take too mach time)
2) simuleted output of our sensor for each input profiel.
3) traning NN for decoding input profiel by the simuleted output.
4) saving model prediction 

See_results.ipynb is exampel with a litl result analis. in it you can check predictions of our net for different sensor structure. And loook to the worst, the best and medium predictions

in bous noutbooks you need to chenge dir_of_data on the part to folder with data on your Google Grive.
