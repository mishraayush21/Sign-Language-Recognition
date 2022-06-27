package com.example.sign_lang;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.PackageManagerCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.nfc.Tag;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class CameraActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private Mat mRgba;
    private Mat mGray;
    private Button okBtn;
    private Button spaceBtn;
    private Button delBtn;
    private TextView textView;
    private StringBuffer text;


    private CameraBridgeViewBase mOpenCvCameraView;
    private objectDetector objDetector;
    private BaseLoaderCallback mLoaderCallback =new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS){
                    Log.i("MainActivity ","OpenCv Is loaded");
                    mOpenCvCameraView.enableView();
                }
            else
                {
                    super.onManagerConnected(status);

                }

            }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        int MY_PERMISSIONS_REQUEST_CAMERA=0;
        if(ContextCompat.checkSelfPermission(CameraActivity.this, Manifest.permission.CAMERA)
        == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(CameraActivity.this, new String[]{Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }


        setContentView(R.layout.activity_camera);
        okBtn = findViewById(R.id.okBtn);
        spaceBtn = findViewById(R.id.spaceBtn);
        delBtn = findViewById(R.id.delBtn);
        textView = findViewById(R.id.text_output);
        textView.setMovementMethod(new ScrollingMovementMethod());
        text = new StringBuffer("");
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        // Copy-pasting the model.tflite and label in assets folder
        // We trained model on input size =300 for hand
        // input size = 96 for sign lang
        //input size = 320 for custom

        try {
            objDetector=new objectDetector(getAssets(),"hand_model.tflite", "model.tflite","custom_label_hand.txt",96,300);
            Log.d("MainActivity","Model is successfully loaded");
        }
        catch (Exception e)
        {
            Log.d("MainActivity", "Errors occurred");
            e.printStackTrace();
        }


        okBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (!text.equals("null") && objDetector.sign_val.length() != 0 ) {
                    text.append(objDetector.sign_val);
                    textView.setText(text);

                }
            }
        });

        delBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(text.length()!=0)
                {
                    text.deleteCharAt(text.length()-1);
                    textView.setText(text);

                }
            }
        });

        spaceBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(text.length()!=0 && text.charAt(text.length()-1) !=' ' )
                {
                    text.append(" ");
                    textView.setText(text);
                }
            }
        });



    }

        @Override
        protected void onResume() {
            super.onResume();
            if (OpenCVLoader.initDebug()){
                    //if load success
                Log.d("MainActivity ","Opencv initialization is done");
                mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            }
            else{
                Log.d("MainActivity ","Opencv is not loaded. try again");
                OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION,this,mLoaderCallback);
            }
        }

        @Override
        protected void onPause() {
            super.onPause();
            if (mOpenCvCameraView !=null){
                mOpenCvCameraView.disableView();
            }
        }

        public void onDestroy(){
            super.onDestroy();
            if(mOpenCvCameraView !=null){
                mOpenCvCameraView.disableView();
            }

        }


    public void onCameraViewStarted(int width ,int height){
        mRgba=new Mat(height,width, CvType.CV_8UC4);
        mGray =new Mat(height,width,CvType.CV_8UC1);
    }
    public void onCameraViewStopped(){
        mRgba.release();
    }
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
        mRgba=inputFrame.rgba();
        mGray=inputFrame.gray();

        Mat out=new Mat();
        out=objDetector.recognizeImage(mRgba);

        return out;
    }






}