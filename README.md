# Identifying Regions Of Face in Images

This project is designed to identify faces in images by using Haar Cascades along with our own methods for reducing false 
positives and false negatives. To reduce false positives, we introduced a method to check for skin in a detected region. To
reduce the number of false negatives the images are upscaled before we run our Cascades, this gives the cascades a greater
chance of picking up faces as they are much larger.

# Running the Application

-   Download or clone the repository to a folder of your choice
-   Using the command line or anaconda prompt navigate to this folder
-   Type in the following command

```sh
$ python FaceDetection.py
```

# Results

Here is our first test image given to us.
![Graduation No Face Detection](https://i.imgur.com/VRRZ8WI.jpg)

And here are the results of running the application on the image.
![Graduation Tested](https://i.imgur.com/2anlZzE.jpg)

Here is an additional image of the Manchester United football team we used for testing.
![Image of Yaktocat](https://i.imgur.com/v3xxZCd.jpg)

And here are our results before checking for false positives.
![Tested Manuited](https://i.imgur.com/mSbtVOw.jpg)

And here are our results with the complete application.
![Pogba Dab](https://i2-prod.mirror.co.uk/incoming/article10494517.ece/ALTERNATES/s615/Manchester-Uniteds-Paul-Pogba-celebrates-winning-the-Europa-League.jpg)

# License 

MIT

