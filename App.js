import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { Camera } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';
import { bundleResourceIO, cameraWithTensors } from '@tensorflow/tfjs-react-native';


// declare camera.
const TensorCamera = cameraWithTensors(Camera);
export default function App() { 
  // declare emotion model state.
  let emotionModel;
  let faceDetector;
  const emotionTypes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
  // emotion model.
  const EmotionModel = async () => {
    const modelJson =  require('./assets/models/emotion-large.json');
    const modelWeights = require('./assets/models/emotion-large.bin');
    const model = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
    return model;
  };
  // on mount.
  useEffect(() => {
    (async () => {
      // ready tf.
      await tf.ready();
      // set the emotion model.
      emotionModel = await EmotionModel();
      faceDetector = await blazeface.load();
    })();
  });
  const handleStream =  (images, updateCameraPreview, gl) => {
    const loop = async () => {
      // get each frame 
      const imageTensor = images.next().value
      // check if the tensor is undefine and it detect faces.
      if (imageTensor !== undefined) {
        let currentTensor = imageTensor;
        // see if the imageTensor is existed.
        if (currentTensor && faceDetector && emotionModel) {
          // get the face from the image.
          let faces = await faceDetector.estimateFaces(currentTensor, false)
          // crop the faces.
          const facesTensors = faces.map(face => {
            return tf.image.cropAndResize(
              currentTensor.reshape([1, currentTensor.shape[0], currentTensor.shape[1], currentTensor.shape[2]]),
              [[
                face.topLeft[1] * currentTensor.shape[1], // y * width
                face.topLeft[0] * currentTensor.shape[0],
                face.bottomRight[1] * currentTensor.shape[1],
                face.bottomRight[0] * currentTensor.shape[0]
              ]],
              [faces.length],
              [64, 64],
              'bilinear'
            )
          }).map(face => {
            // grayscale the image.
            // the vector to normalize the channel in the order [red, green, blue].
            const grayMatrix = [0.2989, 0.5870, 0.1140];
            // devide the face into 3 differnce color channel.
            const [red, green, blue] = tf.split(face, 3, 3);
            face.dispose();
            // normalize the channels with the gray matrix.
            const redNorm = tf.mul(red, grayMatrix[0]);
            const greenNorm = tf.mul(green, grayMatrix[1]);
            const blueNorm = tf.mul(blue, grayMatrix[2]);
            // dispose the normal channels.
            red.dispose();
            green.dispose();
            blue.dispose();
            // merge the new channel to a new grayFace.
            const grayFace = tf.tidy(() => {
              return tf.addN([redNorm, greenNorm, blueNorm]).sub(0.5).mul(2);
            })
            // dispose each individual channels.
            redNorm.dispose();
            greenNorm.dispose();
            blueNorm.dispose();
            // return the matrix;
            return grayFace;
          })
          console.log('facesTensors', facesTensors)

          if (facesTensors.length > 0 && emotionModel) {
            // only return the text prdiction with it persentage.
            const emotionPredictions = facesTensors.map(faceTensor => {
              // call the emotion prediction and get it data.
              const emotionProb = emotionModel
                .predict(faceTensor)
                .dataSync()
              ;
              // return the percentage and and it type.
              const maxProp = Math.max(...emotionProb);
              console.log('maxProp', maxProp)
              return {
                emotion: emotionTypes[emotionProb.indexOf(maxProp)],
                percentage: maxProp
              }
            });
            console.log(emotionPredictions)
          }
        }
      }

      updateCameraPreview();
      gl.endFrameEXP();
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
        // autorender={true}
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
