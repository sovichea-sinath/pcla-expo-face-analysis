import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { Camera } from 'expo-camera';
import * as FaceDetector from 'expo-face-detector';
import * as tf from '@tensorflow/tfjs';
import { fetch, decodeJpeg, bundleResourceIO } from '@tensorflow/tfjs-react-native';

export default function App() {
  // declate the permission state.
  const [hasPermissionState, setHasPermissionState] = useState(null);
  // TEST state.
  const [facesState, setFacesState] = useState(null)
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
      // take the photo when the face is detected.
      const photo = await this.camera.takePictureAsync();
      const { uri } = photo;
    }
  }

  // render the mobile page accordingly to the permission state.
  if (hasPermissionState === null) {
    return <View />;
  }
  if (hasPermissionState === false) {
    return <Text>No access to camera</Text>;
  }
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
