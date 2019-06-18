import numpy as np
import tensorflow as tf
from PIL import Image
import cv2 as cv

# from utils import label_map_util
# from utils import visualization_utils as vis_util
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        # #TODO load classifier
        # Model preparation
        PATH_TO_FROZEN_GRAPH = 'light_classification/model/frozen_inference_graph_sim.pb'
        # PATH_TO_LABELS = 'light_classification/model/light.pbtxt'
        
        # load frozen tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        #self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        self.category_index = {1: {'id': 1, 'name': 'Green'}, 2: {'id': 2, 'name': 'Red'},
                               3: {'id': 3, 'name': 'Yellow'}, 4: {'id': 4, 'name': 'off'}}
        
        self.image_result_show = False

        self.light_thresh = 0.9
    # def load_image_into_numpy_array(self, image):
    #     (im_width, im_height) = image.size
    #     return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def run_inference_for_single_image(self,image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

                # needn't        
                # if 'detection_masks' in tensor_dict:
                #     # The following processing is only for single image
                #     detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                #     detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                #     # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                #     real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                #     detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                #     detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                #     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                #         detection_masks, detection_boxes, image.shape[1], image.shape[2])
                #     detection_masks_reframed = tf.cast(
                #         tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                #     # Follow the convention by adding back the batch dimension
                #     tensor_dict['detection_masks'] = tf.expand_dims(
                #         detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                        feed_dict={image_tensor: image})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.int64)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                # if 'detection_masks' in output_dict:
                #     output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # image_np = self.load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)

        # Actual detection.
        output_dict = self.run_inference_for_single_image(image_np_expanded, self.detection_graph)
        # print(output_dict['num_detections'])
        # print(output_dict['detection_classes'])
        # print(output_dict['detection_scores'])
        # print(output_dict['detection_boxes'])

        # Show classified result image
        if self.image_result_show:
            img_width,img_height,depth = image.shape
            font = cv.FONT_HERSHEY_COMPLEX
            for i in range(output_dict['num_detections']):
                top_y = output_dict['detection_boxes'][i][0] * img_width
                top_x = output_dict['detection_boxes'][i][1] * img_height
                botom_y = output_dict['detection_boxes'][i][2] * img_width
                botom_x = output_dict['detection_boxes'][i][3] * img_height
                score = output_dict['detection_scores'][i]
                class_name = self.category_index[output_dict['detection_classes'][i]]['name']
                cv.rectangle(image, (int(top_x), int(top_y)), (int(botom_x), int(botom_y)), (0, 0, 255), thickness=2)
                cv.putText(image, class_name + ' ' + str(score), (int(top_x), int(top_y)), font, 0.8,(255, 255, 255), 1)
            cv.namedWindow('result', cv.WINDOW_NORMAL)
            cv.imshow('result', image)
            cv.waitKey(1)
        
        # Get red light max score
        red_score_max = 0
        for i in range(output_dict['num_detections']):
            if output_dict['detection_classes'][i] == 2:
                if output_dict['detection_scores'][i] > red_score_max:
                    red_score_max = output_dict['detection_scores'][i]
        if red_score_max > self.light_thresh:
            current_light = TrafficLight.RED
        else:
            current_light = TrafficLight.GREEN
        print(current_light)    
        return current_light
