import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.category.DefaultCategoryDataset;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

public class DDS extends JFrame {

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }  // Load OpenCV

    private VideoCapture cap;
    private List<Double> radius = new ArrayList<>();
    private boolean blink = false;
    private CascadeClassifier eyeCascade;

    public DDS() {
        setTitle("DDS");
        setSize(500, 500);
        setLayout(null);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);

        JLabel title = new JLabel("Drug Detection System");
        title.setBounds(50, 20, 400, 40);
        title.setFont(new Font("Arial", Font.BOLD, 20));
        title.setForeground(Color.RED);
        add(title);

        JLabel info = new JLabel("<html>Drug abuse can affect several aspects<br>"
                + "of a person's physical and psychological health.<br>"
                + "We can detect this by observing pupil dilation after drug intake.</html>");
        info.setBounds(50, 70, 400, 100);
        add(info);

        JButton startButton = new JButton("Start the test for dilation");
        startButton.setBounds(115, 340, 250, 40);
        add(startButton);

        JButton exitButton = new JButton("Exit");
        exitButton.setBounds(215, 400, 100, 40);
        add(exitButton);

        startButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                JOptionPane.showMessageDialog(null, "The Test Will Begin Now...");
                dilation();
            }
        });

        exitButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                JOptionPane.showMessageDialog(null, "You will be leaving this GUI now.");
                System.exit(0);
            }
        });

        // Load the eye cascade classifier
        eyeCascade = new CascadeClassifier("haarcascade_eye.xml");
        cap = new VideoCapture("Video.mp4");
    }

    // Method for pupil dilation detection
    public void dilation() {
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        try {
            if (cap.isOpened()) {
                Mat img = new Mat();
                while (cap.read(img)) {
                    Mat gray = new Mat();
                    Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);
                    MatOfRect eyes = new MatOfRect();
                    eyeCascade.detectMultiScale(gray, eyes);

                    if (eyes.toArray().length > 0) {
                        if (blink) {
                            blink = false;
                        }
                        Imgproc.putText(img, "Detecting for Dilation...", new Point(10, 30), Imgproc.FONT_HERSHEY_COMPLEX_SMALL, 1, new Scalar(0, 255, 0), 2);
                        for (Rect eye : eyes.toArray()) {
                            Imgproc.rectangle(img, eye, new Scalar(0, 255, 0), 2);
                            Mat roiGray = new Mat(gray, eye);
                            Mat roiImg = new Mat(img, eye);
                            Mat blur = new Mat();
                            Imgproc.GaussianBlur(roiGray, blur, new Size(5, 5), 10);
                            Mat erosion = new Mat();
                            Imgproc.erode(blur, erosion, kernel, new Point(-1, -1), 2);
                            Mat thresh = new Mat();
                            Imgproc.threshold(erosion, thresh, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

                            // Hough Circles for detecting pupil
                            Mat circles = new Mat();
                            Imgproc.HoughCircles(roiGray, circles, Imgproc.HOUGH_GRADIENT, 6, 1000, 50, 30, 1, 40);
                            if (!circles.empty()) {
                                for (int i = 0; i < circles.cols(); i++) {
                                    double[] circle = circles.get(0, i);
                                    int r = (int) Math.round(circle[2]);
                                    if (r > 20) {
                                        radius.add(r * 0.2645833333);  // Convert pixels to mm
                                    }
                                }
                            }
                        }
                    } else {
                        if (!blink) {
                            blink = true;
                            Imgproc.putText(img, "Eye not found", new Point(10, 90), Imgproc.FONT_HERSHEY_COMPLEX_SMALL, 1, new Scalar(0, 0, 255), 2);
                        }
                    }

                    HighGui.imshow("Dilation", img);
                    if (HighGui.waitKey(1) == 'q') {
                        JOptionPane.showMessageDialog(null, "Please Complete the test to show the results...");
                        break;
                    }
                }
            }

            cap.release();
            HighGui.destroyAllWindows();

        } catch (Exception e) {
            e.printStackTrace();
        }

        // Plot the dilation graph
        plotGraph();
    }

    // Plotting pupil dilation graph
    public void plotGraph() {
        // Downsample radius readings (similar to skipping every 10th frame)
        List<Double> filteredRadius = new ArrayList<>();
        for (int i = 0; i < radius.size(); i += 10) {
            filteredRadius.add(radius.get(i));
        }

        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (int i = 0; i < filteredRadius.size(); i++) {
            dataset.addValue(filteredRadius.get(i), "Dilation", Integer.toString(i));
        }

        JFreeChart chart = ChartFactory.createLineChart(
                "Pupil Dilation Over Time",
                "Time (Seconds)",
                "Pupil Dilation (mm)",
                dataset
        );

        ChartPanel chartPanel = new ChartPanel(chart);
        JFrame chartFrame = new JFrame();
        chartFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        chartFrame.setContentPane(chartPanel);
        chartFrame.pack();
        chartFrame.setVisible(true);
    }

    public static void main(String[] args) {
        DDS frame = new DDS();
        frame.setVisible(true);
    }
}
