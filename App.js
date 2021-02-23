import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { Camera } from 'expo-camera';
import * as FaceDetector from 'expo-face-detector';
import * as FileSystem from 'expo-file-system';
import * as tf from '@tensorflow/tfjs';
import { decodeJpeg, bundleResourceIO, grayscale_image } from '@tensorflow/tfjs-react-native';

export default function App() {
  // declate the permission state.
  const [hasPermissionState, setHasPermissionState] = useState(null);
  const [isMountState, setIsMountState] = useState(true)
  // set action when the component is create.
  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestPermissionsAsync();
      setHasPermissionState(status === 'granted');
    })();
  }, []);

  // function to handle the camera.
  const handleFaceDetected = async (obj) => {
    if (obj.faces.length > 0) {
      try {
        // take the photo when the face is detected.
        const photo = await this.camera.takePictureAsync();
        const { uri } = photo;
        console.log('uri', uri)
        // so open as base64, turn to raw, then turn to matrix;
        const imgB64 = await FileSystem.readAsStringAsync(uri, {
          encoding: FileSystem.EncodingType.Base64
        });
        await tf.ready();
        const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
        const raw = new Uint8Array(imgBuffer) ;
        // load.
        let imageTensor = decodeJpeg(raw);
        console.log('imageTensor', imageTensor);
        // resize.
        imageTensor = tf.image.resizeBilinear(imageTensor, [48, 48])
        console.log('imageTensorResized', imageTensor)
        // grayscale. because the fucking library don't have rgb_to_grayscale
        imageTensor = imageTensor.mean(2).toFloat().expandDims(-1);
        console.log('imageTensor grayscale', imageTensor);
      } catch (e) {
        console.log('an error occur during camera stuff.');
        setIsMountState(false);
      }
    }
  }

  // render the mobile page accordingly to the permission state.
  if (hasPermissionState === null) {
    return <View />;
  }
  if (hasPermissionState === false) {
    return <Text>No access to camera</Text>;
  }
  if (isMountState) {
    return (
      <View style={styles.container}>
        <Camera
          style={styles.camera}
          type={Camera.Constants.Type.front}
          ref={ ref => { this.camera = ref } }
          onFacesDetected={e => handleFaceDetected(e)}
          faceDetectorSettings={{
            mode: FaceDetector.Constants.Mode.accurate,
            detectLandmarks: FaceDetector.Constants.Landmarks.none,
            runClassifications: FaceDetector.Constants.Classifications.none,
            minDetectionInterval: 100,
            tracking: true,
          }}
        >

        </Camera>
      </View>
    );
  } else {
    return (
      <View>
        <Text> Camera dismounted! </Text>
      </View>
    )
  }
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
