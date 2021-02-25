import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { Camera } from 'expo-camera';
import * as FaceDetector from 'expo-face-detector';
import * as FileSystem from 'expo-file-system';
import * as tf from '@tensorflow/tfjs';
import * as faceapi from 'face-api.js';
import * as blazeface from '@tensorflow-models/blazeface';
import { bundleResourceIO, cameraWithTensors } from '@tensorflow/tfjs-react-native';


// declare camera.
const TensorCamera = cameraWithTensors(Camera);
export default function App() { 
  // declare emotion model state.
  let emotionModel;
  let faceDetector;
  // emotion model.
  const EmotionModel = async () => {
    const modelJson =  require('./assets/models/emotion_model/model.json');
    const modelWeights = require('./assets/models/emotion_model/group1-shard1of1.bin');
    const model = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
    return model;
  };
  // on mount.
  useEffect(() => {
    (async () => {
      // ready tf.
      await tf.ready();
      console.log('tf is ready')
      // set the emotion model.
      emotionModel = await EmotionModel();
      faceDetector = await blazeface.load();
      console.log('model loaded')
    })();
  });
  const handleStream =  (images) => {
    const loop = async () => {
      // get each frame 
      const imageTensor = images.next().value
      // check if the tensor is undefine and it detect faces.
      if (imageTensor !== undefined) {
        let currentTensor = imageTensor;
        // see if the imageTensor is existed.
        if (currentTensor && faceDetector) {
          // get the face from the image.
          let faces = await faceDetector.estimateFaces(currentTensor, false)
          // crop the faces.
          const facesTensors = faces.map(face => {
            return tf.image.cropAndResize(
              currentTensor.reshape([1,256,256,3]),
              [[face.topLeft[1]/256, face.topLeft[0]/256, face.bottomRight[1]/256, face.bottomRight[0]/256]],
              [faces.length],
              [48, 48],
              'bilinear'
            )
          })
          console.log('facesTensors', facesTensors)

          if (facesTensors.length > 0 && emotionModel) {
            const predictions = facesTensors.map(faceTensor => {
              // greyscale the image.
              // faceTensor = faceTensor.reshape([48, 48, 3]).mean(2).expandDims(2);
              // call the emotion
              return emotionModel.predict(faceTensor);
            });
            console.log(predictions)
          }
        }
      }

      requestAnimationFrame(loop);
    }
    loop();
  }
  // set camera dimension.
  const textureDims = Platform.OS === 'ios' ?
  {
    height: 1920,
    width: 1080,
  } : {
    height: 1200,
    width: 1600,
  };
  return (
    <View style={styles.container}>
      <TensorCamera
        // Standard Camera props
        style={styles.camera}
        type={Camera.Constants.Type.front}
        // Tensor related props
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        resizeHeight={256}
        resizeWidth={256}
        resizeDepth={3}
        onReady={handleStream}
        autorender={true}
      />
    </View>
  );

}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'row',
    margin: 20,
  },
  button: {
    flex: 0.1,
    alignSelf: 'flex-end',
    alignItems: 'center',
  },
  text: {
    fontSize: 18,
    color: 'white',
  },
});
