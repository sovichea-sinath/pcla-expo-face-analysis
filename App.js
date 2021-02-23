import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { Camera } from 'expo-camera';
import * as FaceDetector from 'expo-face-detector';
import * as FileSystem from 'expo-file-system';
import * as tf from '@tensorflow/tfjs';
import { decodeJpeg, cameraWithTensors } from '@tensorflow/tfjs-react-native';

// declare camera.
const TensorCamera = cameraWithTensors(Camera);
export default function App() { 
  // on mount.
  useEffect(() => {
    (async () => {
      await tf.ready();
    })();
  }, []);
  // isFaces state.
  const [facesObj, setFacesObj] = useState({faces: [], image: {}});
  // handle face detection.
  const handleFaceDetected = (obj) => {
    console.log('face detected') 
    setFacesObj(obj)
  }
  // handle the stream of images.
  const handleStream = (images) => {
    const loop = async () => {
      // get each frame 
      console.log(images.next())
      const imageTensor = images.next().value
      // check if the tensor is undefine and it detect faces.
      console.log(facesObj.faces.length)
      if (imageTensor !== undefined && facesObj.faces.length > 0) {
        let currentTensor = imageTensor;
        // greyscale the image.
        currentTensor = currentTensor.mean(2).toFloat().expandDims(-1);
      }


      requestAnimationFrame(loop);
    }
    loop();

    // if (obj.faces.length > 0) {
    //   try {
    //     // take the photo when the face is detected.
    //     const photo = await this.camera.takePictureAsync();
    //     const { uri } = photo;
    //     console.log('uri', uri)
    //     // so open as base64, turn to raw, then turn to matrix;
    //     const imgB64 = await FileSystem.readAsStringAsync(uri, {
    //       encoding: FileSystem.EncodingType.Base64
    //     });
    //     await tf.ready();
    //     const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
    //     const raw = new Uint8Array(imgBuffer) ;
    //     // load.
    //     let imageTensor = decodeJpeg(raw);
    //     console.log('imageTensor', imageTensor);
    //     // resize.
    //     imageTensor = tf.image.resizeBilinear(imageTensor, [48, 48])
    //     console.log('imageTensorResized', imageTensor)
    //     // grayscale. because the fucking library don't have rgb_to_grayscale
    //     imageTensor = imageTensor.mean(2).toFloat().expandDims(-1);
    //     console.log('imageTensor grayscale', imageTensor);
    //   } catch (e) {
    //     console.log('an error occur during camera stuff.');
    //     setIsMountState(false);
    //   }
    // }
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
        resizeHeight={48}
        resizeWidth={48}
        resizeDepth={3}
        // onFacesDetected={handleFaceDetected}
        // faceDetectorSettings={{
        //   mode: FaceDetector.Constants.Mode.accuate,
        //   detectLandmarks: FaceDetector.Constants.Landmarks.none,
        //   runClassifications: FaceDetector.Constants.Classifications.none,
        //   minDetectionInterval: 100,
        //   tracking: false,
        // }}
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
