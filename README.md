# Identifying Regions Of Face in Images

This project is designed to identify faces in images by using Haar Cascades along with our own methods for reducing false 
positives and false negatives. To reduce false positives we introduced a method to check for skin in a detected region. To
reduce the number of false negatives the images are upscaled before we run our Cascades, this gives the cascades a greater
chance of picking up faces as they are much larger.

# Running Application

-   Download or clone the repository to a folder of your choice
-   Using the command line or anaconada prompt navigate to this folder
-   Type in the following command

```sh
$ python FaceDetection.py
```

# Results

Here is our first test image given to us.
![Graduation No Face Detection](https://i.imgur.com/VRRZ8WI.jpg)

And here are the results of running the application on the image.
![Graduation Tested](https://i.imgur.com/2anlZzE.jpg)

Here is an additional image of the Manchester United Football Team we used for testing.
![Image of Yaktocat](https://i.imgur.com/v3xxZCd.jpg)

And here are our results before checking for false positives.
![Tested Manuited](https://i.imgur.com/mSbtVOw.jpg)

And here are our results with the complete application.
![Image of Yaktocat](Link goes here)

# License 

MIT


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>

