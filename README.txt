In this project we try to simulate tactile sensor based on the gerd
of optical microfibers. Now, it is a very first steps.
In our sensor, physical number of the output channel is ~sqrt(n), 
where n is number of pixels in classic sensor. It is grate advantage 
in the robotics field. But of course, reducing numbers of channels 
has a side effect, that you can't decode any input signal univocal. 
The goal of this simulation is to show that for physical input 
(pressure profile, that is possible in robotics task), we can decode it 
pretty well even for very big sensor size (and numbers of pixel as well). 
As physical profile we used superposition of several gauss (15 and 30). 
And try to predict this profile by the outputs of our sensor, using a neural network.

Prediction visualisation can be found in [reports\report.md](reports\report.md) file

Pretty good result for 30 gausses is on the 46c92265c2eb15eaa74491cf3580919414ff53cf commit.

Table for comparing results and quality of prediction is [here](https://studio.iterative.ai/user/korbash1/views/RQC-Robotics-tactile_sensor-vvk6t1pklh?mode=full)

install

The easiest way to run our code, is using Google Colab. In this way you need to put folder with data on your Google Drive.
You can download data here:  https://drive.google.com/drive/folders/1qfujkRPA81V8XF0RwKlAvepwT75hBL8Y?usp=sharing
Then you should open examples in Colab. 
main.ipynb is example in which:
1) generated pressure profiles (you can skip this step because it take too much time)
2) simulated output of our sensor for each input profile.
3) training NN for decoding input profile by the simulated output. (for our configuration you need to have colab pro)
4) saving model prediction 

See_results.ipynb is example with a little result analysis. 
In it you can check predictions of our net for different sensor structure. 
And look to the worst, the best and medium predictions.

In both notebooks you need to put in variable dir_of_data the part to folder with data on your Google Drive.
