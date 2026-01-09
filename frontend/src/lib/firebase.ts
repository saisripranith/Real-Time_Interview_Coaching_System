// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyB_zRF-tLxcfRVWJ_yQaTbxQR1wHC5VrhM",
  authDomain: "interviewr-aa890.firebaseapp.com",
  projectId: "interviewr-aa890",
  storageBucket: "interviewr-aa890.firebasestorage.app",
  messagingSenderId: "1085440636512",
  appId: "1:1085440636512:web:582e333b5038eb764e3505",
  measurementId: "G-WZ3LJN5HZP"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const auth = getAuth(app);
const db = getFirestore(app);

export { app, analytics, auth, db };
