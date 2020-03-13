package mx.uagro.artritisreumatoide;

import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;
import android.provider.Settings;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.FileDescriptor;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Vector;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private int PERMISSION_CODE = 10;

    Button btnFoto;
    Button btnDiagnostico;
    EditText etxtFuerzaUno;
    EditText etxtFuerzaDos;
    EditText etxtEdad;
    TextView txtvIma;

    Uri imageUri;

    private static final int PICK_IMAGE = 100;

    private static final int IMG_SIZE = 150;

    private static final int CNN_OUPUT_SIZE = 1;
    private static final int IAFuEd_OUPUT_SIZE = 1;

    private float activacion = 0;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("OpenCV", "OpenCV cargado exitosamente");
                    Mat imageMat = new Mat();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnFoto = findViewById(R.id.btnImg);
        btnDiagnostico = findViewById(R.id.btnDiagnostico);
        etxtFuerzaUno = findViewById(R.id.etxtFuerzaUno);
        etxtFuerzaDos = findViewById(R.id.etxtFuerzaDos);
        etxtEdad = findViewById(R.id.etxtEdad);
        txtvIma = findViewById(R.id.txtvIma);

        btnFoto.setOnClickListener(this);
        btnDiagnostico.setOnClickListener(this);
    }

    private void openGallery(){
        Intent gallery = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI);
        gallery.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivityForResult(gallery,PICK_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode,resultCode,data);
        if(resultCode==RESULT_OK && requestCode == PICK_IMAGE){

            imageUri = data.getData();

            try {

                predictCnnFromUri(imageUri);

            } catch (IOException e) {

                e.printStackTrace();

            }
        }

    }

    private Bitmap ARGBBitmap(Bitmap img){

        return img.copy(Bitmap.Config.ARGB_8888,true);

    }

    private void predictCnnFromUri(Uri uri) throws IOException {
        String CNN_OUTPUT_NODE = "activation_4/Sigmoid";

        float [] result= new float[CNN_OUPUT_SIZE];
        String [] exit={CNN_OUTPUT_NODE};

        ParcelFileDescriptor parcelFileDescriptor =
                getContentResolver().openFileDescriptor(uri, "r");
        FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
        Bitmap bitmapImage = BitmapFactory.decodeFileDescriptor(fileDescriptor);
        parcelFileDescriptor.close();
        bitmapImage = ARGBBitmap(bitmapImage);

        Mat matImage = new Mat(bitmapImage.getWidth(),bitmapImage.getHeight(),CvType.CV_8UC1);

        Utils.bitmapToMat(bitmapImage,matImage);
        Mat Bgr = new Mat();

        Imgproc.cvtColor(matImage,Bgr,Imgproc.COLOR_RGBA2BGR);

        Size sz = new Size(IMG_SIZE,IMG_SIZE);

        Imgproc.resize(Bgr,Bgr,sz);

        float [] values = flattern(Bgr);

        String CNN_MODEL_PATH = "cnnAR.pb";
        String CNN_INPUT_NODE = "batch_normalization_1_input";

        AssetManager assetManager = getAssets();

        TensorFlowInferenceInterface tfHelper = new TensorFlowInferenceInterface(assetManager,
                CNN_MODEL_PATH);

        tfHelper.feed(CNN_INPUT_NODE,values,1,IMG_SIZE,IMG_SIZE,3);
        tfHelper.run(exit);
        tfHelper.fetch(CNN_OUTPUT_NODE,result);

        String cnn = Float.toString(result[0]);
        activacion = (float) result[0];

        txtvIma.setText(cnn);

    }

    private float[] flattern (Mat img){

        ArrayList<Float> temp = new ArrayList<>();
        Vector<Mat> bgr_planes = new Vector<>();

        Core.split(img,bgr_planes);

        for (int i = 0; i < bgr_planes.get(0).rows(); i++){

            for (int j = 0; j < bgr_planes.get(0).cols(); j++){

                double [] tempBlue = bgr_planes.get(0).get(i,j);
                double tempBlue2 = tempBlue[0];
                float pixelBlue = (float) tempBlue2;

                double [] tempGreen = bgr_planes.get(1).get(i,j);
                double tempGreen2 = tempGreen[0];
                float pixelGreen = (float) tempGreen2;

                double [] tempRed = bgr_planes.get(2).get(i,j);
                double tempRed2 = tempRed[0];
                float pixelRed = (float) tempRed2;

                temp.add(pixelBlue);
                temp.add(pixelGreen);
                temp.add(pixelRed);

            }


        }

        float[] floatArray = new float[temp.size()];

        int i = 0;

        for (Float f : temp) {
            floatArray[i++] = (f != null ? f : Float.NaN);
            System.out.println(f);
        }

        return floatArray;

    }

    private void preDiagnostico(){

        String IAFuEd_OUTPUT_NODE = "activation_3/Sigmoid";
        String IAFuEd_MODEL_PATH = "IAFuEdAR.pb";
        String IAFuEd_INPUT_NODE = "dense_1_input";

        float [] result= new float[IAFuEd_OUPUT_SIZE];
        String [] exit={IAFuEd_OUTPUT_NODE};

        float f1 = Float.valueOf(etxtFuerzaUno.getText().toString());
        float f2 = Float.valueOf(etxtFuerzaDos.getText().toString());
        float edad = Float.valueOf(etxtEdad.getText().toString());

        float [] caracteristicas = new float[]{activacion, f1, f2, edad};

        AssetManager assetManager = getAssets();

        TensorFlowInferenceInterface tfHelper = new TensorFlowInferenceInterface(assetManager,
                IAFuEd_MODEL_PATH);

        tfHelper.feed(IAFuEd_INPUT_NODE,caracteristicas,1,4);

        tfHelper.run(exit);
        tfHelper.fetch(IAFuEd_OUTPUT_NODE,result);

        float diagnostico = (float) result[0];

        AlertDialog.Builder preDiag = new AlertDialog.Builder(this);
        preDiag.setCancelable(true);
        preDiag.setPositiveButton(
                "Aceptar",
                new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        dialog.cancel();
                    }
                });

        etxtFuerzaUno.setText("");
        etxtFuerzaDos.setText("");
        etxtEdad.setText("");
        txtvIma.setText("Imagen");
        activacion = 0;


        if(diagnostico<0.5){

            preDiag.setMessage("Baja Probabilidad de Artritis.");

        }else {

            preDiag.setMessage("Alta Probabilidad de Artritis.");

        }

        AlertDialog alertDiag = preDiag.create();
        alertDiag.show();

    }

    private void checkAll(){

        AlertDialog.Builder verificar = new AlertDialog.Builder(this);
        verificar.setCancelable(true);
        verificar.setPositiveButton(
                "Aceptar",
                new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        dialog.cancel();
                    }
                });

        if(activacion==0){

            verificar.setMessage("Favor de Seleccionar una Imagen");
            AlertDialog alertDiag = verificar.create();
            alertDiag.show();

        }else if (etxtFuerzaUno.length()==0){

            verificar.setMessage("Favor de Ingresar el Valor de Fuerza 1");
            AlertDialog alertDiag = verificar.create();
            alertDiag.show();
        }else if (etxtFuerzaDos.length()==0){

            verificar.setMessage("Favor de Ingresar el Valor de Fuerza 2");
            AlertDialog alertDiag = verificar.create();
            alertDiag.show();

        }else if (etxtEdad.length()==0){

            verificar.setMessage("Favor de Ingresar la Edad");
            AlertDialog alertDiag = verificar.create();
            alertDiag.show();

        }else {

            preDiagnostico();

        }

    }

    @Override
    public void onClick(View v) {
        switch (v.getId()){
            case R.id.btnImg:
                //Comprobar version de Android
                if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
                    if (checkPermissions(Manifest.permission.READ_EXTERNAL_STORAGE)){
                        //ha aceptado el permiso
                        openGallery();
                    }else{
                        if (!shouldShowRequestPermissionRationale(
                                Manifest.permission.READ_EXTERNAL_STORAGE)){
                            requestPermissions(new String[]
                                    {Manifest.permission.READ_EXTERNAL_STORAGE},
                                    PERMISSION_CODE);
                        }else {
                            Toast.makeText(MainActivity.this,
                                    "Por favor aactive los permisos",
                                    Toast.LENGTH_LONG).show();
                            Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
                            intent.addCategory(Intent.CATEGORY_DEFAULT);
                            intent.setData(Uri.parse("package:"+getPackageName()));
                            intent.addFlags(Intent.FLAG_ACTIVITY_EXCLUDE_FROM_RECENTS);
                            startActivity(intent);
                        }
                    }
                }else {
                    olderVersions();
                }
                break;

            case R.id.btnDiagnostico:
                checkAll();
                break;
        }
    }

    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "No se encontro la libreria de OpenCV");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_2, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "Libreria OpenCV encontrada");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private void olderVersions(){
        if (checkPermissions(Manifest.permission.READ_EXTERNAL_STORAGE)){
            openGallery();
        }else {
            Toast.makeText(MainActivity.this,"No has permitido el acceso",
                    Toast.LENGTH_LONG).show();
        }
    }

    private boolean checkPermissions(String permission){
        int result = this.checkCallingOrSelfPermission(permission);
        return result == PackageManager.PERMISSION_GRANTED;
    }
}

