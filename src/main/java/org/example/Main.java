package org.example;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class Main {

    public static INDArray preprocessImage(String imagePath) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(128, 128, 3);
        INDArray image = loader.asMatrix(new File(imagePath));
        image.divi(255);

        return image;
    }

    public static BufferedImage rotateImage(BufferedImage image, double angle) {
        double radians = Math.toRadians(angle);
        int width = image.getWidth();
        int height = image.getHeight();

        BufferedImage rotatedImage = new BufferedImage(width, height, image.getType());
        Graphics2D g2d = rotatedImage.createGraphics();

        AffineTransform at = new AffineTransform();
        at.rotate(radians, width / 2.0, height / 2.0);
        g2d.setTransform(at);
        g2d.drawImage(image, 0, 0, null);
        g2d.dispose();

        return rotatedImage;
    }

    public static void classifyImagesInFolder(String imgFolder, String outputFolder, MultiLayerNetwork model) throws IOException {
        File folder = new File(imgFolder);
        File outputDir = new File(outputFolder);

        if (!outputDir.exists()) {
            outputDir.mkdirs(); // Create the output directory if it doesn't exist
        }

        for (File imgFile : folder.listFiles()) {
            if (imgFile.getName().endsWith(".jpg")) {
                String imgPath = imgFile.getAbsolutePath();
                INDArray image = preprocessImage(imgPath);
                image.permutei(0, 2, 3, 1);

                INDArray output = model.output(image);
                int predictedClass = output.argMax(1).getInt(0);
                double confidence = output.getDouble(predictedClass) * 100;

                Map<Integer, String> categoryLabels = new HashMap<>();
                categoryLabels.put(0, "left90");
                categoryLabels.put(1, "noRotation");
                categoryLabels.put(2, "right90");
                categoryLabels.put(3, "rotate180");

                String categoryLabel = categoryLabels.get(predictedClass);
                System.out.printf("IMG: %s, Predicted Type: %s, Confidence: %.2f%%%n",
                        imgFile.getName(), categoryLabel, confidence);

                // Load the original image
                BufferedImage originalImage = ImageIO.read(imgFile);
                BufferedImage rotatedImage = null;

                // Rotate the image based on the prediction
                switch (predictedClass) {
                    case 0: // left90
                        rotatedImage = rotateImage(originalImage, -90);
                        break;
                    case 1: // noRotation
                        rotatedImage = originalImage; // No rotation needed
                        break;
                    case 2: // right90
                        rotatedImage = rotateImage(originalImage, 90);
                        break;
                    case 3: // rotate180
                        rotatedImage = rotateImage(originalImage, 180);
                        break;
                }

                // Save the rotated image in the specified output folder
                if (rotatedImage != null) {
                    String rotatedImagePath = Paths.get(outputFolder, imgFile.getName().replace(".jpg", "_rotated.jpg")).toString();
                    ImageIO.write(rotatedImage, "jpg", new File(rotatedImagePath));
                    System.out.printf("Rotated image saved at: %s%n", rotatedImagePath);
                }
            }
        }
    }

    public static void main(String[] args) throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {

        System.out.println(args[0]);
        System.out.println(args[1]);
        System.out.println(args[2]);

        String imgFolder = args[0]; //path to image folder
        String outputFolder = args[1]; //path to rotated image folder

        // Load the model
        File modelFile = new File(args[2]); //model
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(String.valueOf(modelFile), false);
        classifyImagesInFolder(imgFolder, outputFolder, model);
    }
}
