import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, Dimensions, TouchableOpacity } from 'react-native';
import Svg, { Circle, Rect, G, Line} from 'react-native-svg';
import { Camera } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';
import { bundleResourceIO, cameraWithTensors } from '@tensorflow/tfjs-react-native';


// declare camera.
const TensorCamera = cameraWithTensors(Camera);
const windowWidth = Dimensions.get('window').width;
const windowHeight = Dimensions.get('window').height;
// resize the camera display box
const resizedWidth = (windowWidth - windowWidth % 4);
const resizedHeight = (windowHeight - windowHeight % 4);
export default function App() { 
  // declare emotion model state.
  let genderModel;
  let emotionModel;
  let faceDetector;
  const [facesState, setFaceState] = useState([]);
  const emotionTypes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
  // emotion model.
  const EmotionModel = async () => {
    const modelJson =  require('./assets/models/emotion-large.json');
    const modelWeights = require('./assets/models/emotion-large.bin');
    const model = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
    return model;
  };
  // age model
  const GenderModel = async () => {
    const modelJson =  require('./assets/models/gender.json');
    const modelWeights = require('./assets/models/gender.bin');
    const model = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
    return model;
  }
  // on mount.
  useEffect(() => {
    (async () => {
      // ready tf.
      await tf.ready();
      // set the all models.
      genderModel = await GenderModel();
      // emotionModel = await EmotionModel();
      faceDetector = await blazeface.load();
      console.log('models loaded')
    })();
  }, []);

  // draw the bounding box.
  const render_boundingBoxes = () => {
    const faceBoxes = facesState.map((face, i) => {
      const { topLeft, bottomRight } = face;
      console.log(`origin x: ${topLeft[0]}, y: ${topLeft[1]}`)
      return (
        <G key={`facebox_${i}`}>
          <Rect
            x={topLeft[0]}
            y={topLeft[1]}
            fill={'red'}
            fillOpacity={0.2}
            width={(bottomRight[0] - topLeft[0])}
            height={(bottomRight[1] - topLeft[1])}
          />
        </G>
      )
    });
  
      return <Svg height='100%' width='100%'
        viewBox={`0 0 ${resizedWidth} ${resizedHeight}`}
        scaleX={-1}
        scaleY={-1}
        >
          {faceBoxes}
        </Svg>;
  };

  // handle the stream.
  const handleStream =  (images, updateCameraPreview, gl) => {
    const loop = async () => {
      // get each frame 
      const imageTensor = images.next().value
      // check if the tensor is undefine and it detect faces.
      if (imageTensor !== undefined) {
        let currentTensor = imageTensor;
        // see if the imageTensor is existed.
        if (currentTensor && faceDetector) {
          // get the face from the image.
          let faces = await faceDetector.estimateFaces(currentTensor, false);
          if(faces.length !== 0) console.log(faces)
          // set the face state.
          setFaceState(faces);
          // crop the faces.
          const facesTensors = faces//.map(face => {
          //   return tf.image.cropAndResize(
          //     currentTensor.reshape([1, currentTensor.shape[0], currentTensor.shape[1], currentTensor.shape[2]]),
          //     [[
          //       face.topLeft[1] * currentTensor.shape[1], // y * width
          //       face.topLeft[0] * currentTensor.shape[0],
          //       face.bottomRight[1] * currentTensor.shape[1],
          //       face.bottomRight[0] * currentTensor.shape[0]
          //     ]],
          //     [faces.length],
          //     [64, 64],
          //     'bilinear'
          //   )
          /*})*/.map(face => {
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
            // grayscale the image
            const grayscale = tf.addN([redNorm, greenNorm, blueNorm]);
            redNorm.dispose();
            greenNorm.dispose();
            blueNorm.dispose();
            const normalize = tf.tidy(() => grayscale.sub(0.5).mul(2));
            grayscale.dispose();
            return normalize;
            // merge the new channel to a new grayFace.
            // const grayFace = tf.tidy(() => {
            //   return tf.addN([redNorm, greenNorm, blueNorm]).sub(0.5).mul(2);
            // })
            // // dispose each individual channels.
            // redNorm.dispose();
            // greenNorm.dispose();
            // blueNorm.dispose();
            // return the matrix;
            // return grayFace;
          })
          // console.log('facesTensors', facesTensors)

          // if (facesTensors.length > 0 && emotionModel) {
          //   // only return the text prdiction with it persentage.
          //   const emotionPredictions = facesTensors.map(faceTensor => {
          //     // call the emotion prediction and get it data.
          //     const emotionProb = emotionModel
          //       .predict(faceTensor)
          //       .dataSync()
          //     ;
          //     // return the percentage and and it type.
          //     const maxProp = Math.max(...emotionProb);
          //     // console.log('maxProp', maxProp)
          //     return {
          //       emotion: emotionTypes[emotionProb.indexOf(maxProp)],
          //       percentage: maxProp
          //     }
          //   });
          //   console.log(emotionPredictions)
          // }

          // predict the age.
          if (facesTensors.length > 0 && genderModel) {
            // only return the text prdiction with it persentage.
            const genderPredictions = facesTensors.map(faceTensor => {
              // call the emotion prediction and get it data.
              const genderProb = genderModel
                .predict(faceTensor)
                .dataSync()
              ;
              return genderProb[0] > genderProb[1] ?
                {
                  age: 'female',
                  percentage: genderProb[0]
                } : {
                  age: 'male',
                  percentage: genderProb[1]
                }
              ;

              // return the percentage and and it type.
              // const maxProp = Math.max(...emotionProb);
              // // console.log('maxProp', maxProp)
              // return {
              //   emotion: emotionTypes[emotionProb.indexOf(maxProp)],
              //   percentage: maxProp
              // }
            });
            console.log(genderPredictions)
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
    <View style={styles.page}>
      {/* camera view. */}
      <View style={styles.cameraContainer}>
        <TensorCamera
          // Standard Camera props
          style={styles.camera}
          type={Camera.Constants.Type.front}
          // Tensor related props
          cameraTextureHeight={resizedHeight}
          cameraTextureWidth={resizedWidth}
          resizeHeight={resizedHeight}
          resizeWidth={resizedWidth}
          resizeDepth={3}
          onReady={handleStream}
          // autorender={true}
        />
      </View>

      {/* the svg (bounding box) View */}
      <View style={styles.svgContainer}>
        { render_boundingBoxes() }
      </View>
    
    </View>
  );

}

const styles = StyleSheet.create({
  page: {
    width:'100%',
    height: '100%'
    // width: resizedWidth,
    // height: resizedHeight
  },
  cameraContainer: {
    display: 'flex',
    // flex: 1,
    // flexDirection: 'column',
    // justifyContent: 'center',
    // alignItems: 'center',
    width: '100%',
    height: '100%',
    // backgroundColor: '#fff',
  },
  camera: {
    // flex: 1,
    position:'absolute',
    left: 0,
    top: 0,
    width: '100%',
    height: '100%',
    // width: resizedWidth,
    // height: resizedHeight,
    zIndex: 1,
  },
  svgContainer: {
    position: 'absolute',
    left: 0,
    top: 0,
    width: '100%',
    height: '100%',
    zIndex: 20,
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
