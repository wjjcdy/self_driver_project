^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package tl_detector 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2019-6-22
* Add traffic light model which choiced based on traffic_light_config.yaml.

2019-6-18
* Add getting only classfied red max score 
* Add return red light state only if max score bigger than thresh by set

2019-6-17
* Add get_light status from the classfier of the light image in tl_detector.py 
* Modified process classfying each 3 images 
* Add light classfier function for image from tensorflow object detection tutorial
* Add tensorflow object detection API training model 
* Add light classfier result image imshow

2019-6-15
* Delete create .csv file
* Change save .png file name

2019-6-7
-------------------
* Add receive a traffic light picture and save png file used for check and deeplearning data
* Add receive traffic state save .csv file for deeplearning data


2019-6-5
-------------------
* Modified the closest function, old code just find the closet point, did not care the need front the car
* Optimized the latency by putting the get_stopline in init not image callback

2019-6-4
-------------------
* Add light status callback function which was sent by simulator just test first
* Add find closest point function used KDtree
* Add find closest the traffic light function based on the position which the car locates on the waypoints
 






