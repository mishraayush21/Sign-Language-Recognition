package com.example.sign_lang;


import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class objectDetector {

    private Interpreter interpreter; //Need to add dependencies of tensorflow
    //used to label model and predict

    private Interpreter interpreter2;
    //interpreter for sign_lang_model

    private List<String> labels;

    private int INPUT_SIZE ;
    private int CLASSIFICATION_INPUT_SIZE;

    private GpuDelegate gpuDelegate;//used to initialize gpu in app
    private int height=0, width=0;
    public String sign_val = "";
    public objectDetector(AssetManager assetManager, String modelPath, String classification_model , String labelPath,int classification_inp_size, int input_size) throws IOException {
       INPUT_SIZE = input_size;
       CLASSIFICATION_INPUT_SIZE = classification_inp_size;
       Interpreter.Options options = new Interpreter.Options();
       gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4);//no. of core of processor in phone = no. of threads
        // loading model
        interpreter=new Interpreter(loadModelFile(assetManager,modelPath),options);
        // load labelmap
        labels=loadLabelList(assetManager,labelPath);

        Interpreter.Options options2 = new Interpreter.Options();
        options2.setNumThreads(2);//no. of core of processor in phone = no. of threads
        interpreter2=new Interpreter(loadModelFile(assetManager,classification_model),options2);

    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> list = new ArrayList<>();
        BufferedReader br = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));

        String line;
        while ((line = br.readLine()) != null)
            list.add(line);

        br.close();
        return list;

    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath)throws IOException
    {
        AssetFileDescriptor fileDescriptor=assetManager.openFd(modelPath);
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset =fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);

    }

    public Mat recognizeImage(Mat matImg)
    {
        sign_val = "";
        Mat rotated_mat_image=new Mat();

        Mat a=matImg.t();
        Core.flip(a,rotated_mat_image,1);
        // Release mat
        a.release();


        Bitmap bitmap=null;
        bitmap=Bitmap.createBitmap(rotated_mat_image.cols(),rotated_mat_image.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotated_mat_image,bitmap);
        height=bitmap.getHeight();
        width=bitmap.getWidth();

        Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);

        // convert bitmap to bytebuffer as model input should be in it
        ByteBuffer byteBuffer=convertBitmapToByteBuffer(scaledBitmap);

        Object[] input=new Object[1];
        input[0]=byteBuffer;

        Map<Integer,Object> output_map=new TreeMap<>();
        // we are not going to use this method of output
        // instead we create treemap of three array (boxes,score,classes)

        float[][][]boxes =new float[1][10][4];
        // 10: top 10 object detected
        // 4: there coordinate in image

        float[][] scores=new float[1][10];
        float[][] classes=new float[1][10];

        output_map.put(0,boxes);
        output_map.put(1,classes);
        output_map.put(2,scores);

        //predicting
        interpreter.runForMultipleInputsOutputs(input,output_map);

        Object value=output_map.get(0);
        Object Object_class=output_map.get(1);
        Object score=output_map.get(2);

        for (int i=0;i<10;i++){
            float class_value=(float) Array.get(Array.get(Object_class,0),i);
            float score_value=(float) Array.get(Array.get(score,0),i);
            // define threshold for score

            // we can change threshold according to your model
            if(score_value>0.5){
                Object box1=Array.get(Array.get(value,0),i);
                //  multiplying it with Original height and width of frame

                float y1=(float) Array.get(box1,0)*height;
                float x1=(float) Array.get(box1,1)*width;
                float y2=(float) Array.get(box1,2)*height;
                float x2=(float) Array.get(box1,3)*width;


                y1 = Math.max(0,y1);
                x1 = Math.max(0,x1);
                y2 = Math.min(height,y2);
                x2 = Math.min(width,x2);

                float w1 = x2-x1;
                float h1 = y2-y1;

                //x1,y1 starting point of hand and (x2,y2) end point

                //cropping hand image from original img
                Rect cropped_roi = new Rect((int)x1, (int)y1, (int)w1, (int)h1);

                Mat cropped = new Mat(rotated_mat_image,cropped_roi).clone();
                //convert cropped mat to bitmap
                Bitmap bitmap1= null;
                bitmap1 = Bitmap.createBitmap(cropped.cols(),cropped.rows(),Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(cropped,bitmap1);

                Bitmap scaledBitmap1 = Bitmap.createScaledBitmap(bitmap1, CLASSIFICATION_INPUT_SIZE,CLASSIFICATION_INPUT_SIZE,false);
                ByteBuffer byteBuffer1 = convertBitmapToByteBuffer1(scaledBitmap1);


                float [][] output_class_value = new float[1][1];
                interpreter2.run(byteBuffer1, output_class_value);

                sign_val = getAlphabets(output_class_value[0][0]);





                Imgproc.putText(rotated_mat_image,""+sign_val,new Point(x1+10,y1+40),2,1.5,new Scalar(255, 255, 255, 255),2);

                // draw rectangle in Original frame //  starting point    // ending point of box  // color of box  //thickness
                Imgproc.rectangle(rotated_mat_image,new Point(x1,y1),new Point(x2,y2),new Scalar(0, 255, 0, 255),2);
                // write text on frame
                // string of class name of object  // starting point       // color of text    // size of text


            }

        }
        Mat b=rotated_mat_image.t();
        Core.flip(b,matImg,0);
        b.release();

        return matImg;

    }

    private String getAlphabets(float v) {
        Log.i("Detecting ", ""+v);
        String val = "";
        if(v>=-0.5 && v<0.5)
            val = "A";
        else if(v>=0.5 && v<1.5)
            val = "B";
        else if(v>=1.5 && v<2.5)
            val = "C";
        else if(v>=2.5 && v<3.5)
            val = "D";
        else if(v>=3.5 && v<4.5)
            val = "E";
        else if(v>=4.5 && v<5.5)
            val = "F";
        else if(v>=5.5 && v<6.5)
            val = "G";
        else if(v>=6.5 && v<7.5)
            val = "H";
        else if(v>=7.5 && v<8.5)
            val = "I";
        else if(v>=8.5 && v<9.5)
            val = "J";
        else if(v>=9.5 && v<10.5)
            val = "K";
        else if(v>=10.5 && v<11.5)
            val = "L";
        else if(v>=11.5 && v<12.5)
            val = "M";
        else if(v>=12.5 && v<13.5)
            val = "N";
        else if(v>=13.5 && v<14.5)
            val = "O";
        else if(v>=14.5 && v<15.5)
            val = "P";
        else if(v>=15.5 && v<16.5)
            val = "Q";
        else if(v>=16.5 && v<17.5)
            val = "R";
        else if(v>=17.5 && v<18.5)
            val = "S";
        else if(v>=18.5 && v<19.5)
            val = "T";
        else if(v>=19.5 && v<20.5)
            val = "U";
        else if(v>=20.5 && v<21.5)
            val = "V";
        else if(v>=21.5 && v<22.5)
            val = "W";
        else if(v>=22.5 && v<23.5)
            val = "X";
        else if(v>=23.5 && v<24.5)
            val = "Y";
        return val;
    }


    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        //some model input should be quant=0  for some quant=1
        ByteBuffer byteBuffer;
        int quant=1;
        int size_images=INPUT_SIZE;
        if(quant==0){
            byteBuffer=ByteBuffer.allocateDirect(1*size_images*size_images*3);
        }
        else {
            byteBuffer=ByteBuffer.allocateDirect(4*1*size_images*size_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;

        // some error
        //now run
        for (int i=0;i<size_images;++i){
            for (int j=0;j<size_images;++j){
                final  int val=intValues[pixel++];
                if(quant==0){
                    byteBuffer.put((byte) ((val>>16)&0xFF));
                    byteBuffer.put((byte) ((val>>8)&0xFF));
                    byteBuffer.put((byte) (val&0xFF));
                }
                else {
                    // paste this
                    byteBuffer.putFloat((((val >> 16) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val >> 8) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val) & 0xFF))/255.0f);
                }
            }
        }
        return byteBuffer;
    }



    private ByteBuffer convertBitmapToByteBuffer1(Bitmap bitmap) {
        //some model input should be quant=0  for some quant=1
        ByteBuffer byteBuffer;
        int quant=1;
        int size_images=CLASSIFICATION_INPUT_SIZE;
        if(quant==0){
            byteBuffer=ByteBuffer.allocateDirect(1*size_images*size_images*3);
        }
        else {
            byteBuffer=ByteBuffer.allocateDirect(4*1*size_images*size_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;

        // some error
        //now run
        for (int i=0;i<size_images;++i){
            for (int j=0;j<size_images;++j){
                final  int val=intValues[pixel++];
                if(quant==0){
                    byteBuffer.put((byte) ((val>>16)&0xFF));
                    byteBuffer.put((byte) ((val>>8)&0xFF));
                    byteBuffer.put((byte) (val&0xFF));
                }
                else {
                    // paste this
                    byteBuffer.putFloat((((val >> 16) & 0xFF)));
                    byteBuffer.putFloat((((val >> 8) & 0xFF)));
                    byteBuffer.putFloat((((val) & 0xFF)));
                }
            }
        }
        return byteBuffer;
    }


}
