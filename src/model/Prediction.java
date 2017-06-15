package model;

import object.AreaObject;
import object.UserObject;
import object.VenueObject;
import utils.Function;
import utils.Utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import java.util.*;

/**
 * Created by tndoan on 4/26/17.
 */
public class Prediction extends Model{

    private HashMap<String, HashMap<String, Double>> gt;

    public Prediction(String uFile, String venueLocFile, String cksFile, double steepness, int k, double scale,
                      boolean isFriend, String friendshipFile, String resultFile, String groundTruthFile)
            throws IOException {
        super(uFile, venueLocFile, cksFile, friendshipFile, k, scale, isFriend, steepness);
        System.out.println("Finish loading");
        long sTime = System.currentTimeMillis();
        readResultFile(resultFile);
        System.out.println("Finish reading result file: " + ((System.currentTimeMillis() - sTime)/1000) + " s");
        sTime = System.currentTimeMillis();
        gt = readGroundTruth(groundTruthFile);
        System.out.println("Finish reading groundtruth file: " + ((System.currentTimeMillis() - sTime)/1000) + " s");
        System.out.println("Groundtruth key set size: " + gt.keySet().size());
    }

    public void printTopK(int[] topk) {
        Arrays.sort(topk);
        int maxTopk = topk[topk.length - 1];
        double count = 0;
        int c = 0; int total = gt.keySet().size();
        Set<String> vSet = new HashSet<>();
        for (String uId : gt.keySet()) {
            Set<String> v = gt.get(uId).keySet();
            for (String vId : v)
                if (venueMap.get(vId) != null)
                    vSet.add(vId);
        }

        for (String userId : gt.keySet())
            if (userMap.get(userId) != null)
                count++;

        System.out.println("vSet :" + vSet.size());

        double[] result = new double[topk.length];

//        gt.keySet().parallelStream().forEach(userId -> {
        for (String userId : gt.keySet()) {
            c++;
            if (userMap.get(userId) == null) // if not in training set, ignore
                return;

//            System.out.println("user id:" + userId);
            // prediction
            List<PairObject> list = Collections.synchronizedList(new ArrayList<>(20000));

            long sTime = System.currentTimeMillis();

//            vSet.parallelStream().forEach(venueId -> {
            for (String venueId : vSet) {
//            for (String venueId : venueMap.keySet()) {
                double pred = calculatePredictedProb(userId, venueId);
                list.add(new PairObject(venueId, pred));
            }
//            });
//            System.out.println("1st part:" + ((System.currentTimeMillis() - sTime)/1000) + " s");

            HashMap<String, Double> gtOfUser = gt.get(userId);
            if (gtOfUser == null)
                return;
            Set<String> groundTruth = gtOfUser.keySet();
            ArrayList<String> topkList = topKVenues(list, maxTopk);

            // compare to groundtruth
            for (int i = 0; i < topk.length; i++) {
                int tk = topk[i];
                Set<String> t = new HashSet<>(topkList.subList(0, tk));
                Set<String> g = new HashSet<>(groundTruth);

                Set<String> intersection = new HashSet<>();
                intersection.addAll(t);
                intersection.retainAll(g);

                synchronized (result) {
                    result[i] += (double) intersection.size() / (double) groundTruth.size();
                }
            }
//            System.out.println("Finish user :" + userId + ":" + ((System.currentTimeMillis() - sTime)/1000) + " s");
            if (c % 1000 == 0) // limit the output. I don't want to flush all
                System.out.println(c + "/" + total);
        }
//        });

        for (int i = 0; i < topk.length; i++) {
            double r = result[i] / count;
            System.out.println("Top " + topk[i] + ":" + r);
        }
    }

    /**
     * calculate the probability of check-in between user and venue
     * @param uId   user id
     * @param vId   venue id
     * @return      probability of check-in between this pair of user and venue
     */
    private double calculatePredictedProb(String uId, String vId) {
        UserObject uo = userMap.get(uId);
        VenueObject vo = venueMap.get(vId);

        double[] uFactor = uo.getFactors();
        AreaObject ao = areaMap.get(vo.getAreaId());
        double[] aFactor = new double[k];

        for (String nId : ao.getSetOfVenueIds()) {
            double[] nFactor = venueMap.get(nId).getFactors();
            aFactor = Function.plus(aFactor, nFactor);
        }
        double result = Math.log(Function.innerProduct(uFactor, aFactor));

        double lhs = Function.innerProduct(uFactor, vo.getFactors());
        result += vo.getNeighbors().parallelStream().mapToDouble(nId -> {
//        for (String nId : vo.getNeighbors()) {
            double[] nFactor = venueMap.get(nId).getFactors();
            double rhs = Function.innerProduct(uFactor, nFactor);
            double diff = lhs - rhs;
            diff = Math.log(Function.logisticFunc(steepness, diff));

//            result += diff;
            return diff;
        }).sum();

//        for (String nId : vo.getNeighbors()) {
//            double[] nFactor = venueMap.get(nId).getFactors();
//            double rhs = Function.innerProduct(uFactor, nFactor);
//            double diff = lhs - rhs;
//            if (isSigmoid)
//                diff = Math.log(Function.sigmoidFunction(diff));
//            else
//                diff = Math.log(Function.tanh1_2(diff));
//
//            result += diff;
//        }

        return result;
    }

    private static ArrayList<String> topKVenues(List<PairObject> orig, int nummax) {
        Collections.sort(orig, new Comparator<PairObject>() {

            @Override
            public int compare(PairObject o1, PairObject o2) {
                if (o1.cks == o2.cks)
                    return 0;
                return o1.cks < o2.cks ? 1 : -1;
            }
        });

        List<PairObject> p = orig;
        if (orig.size() >= nummax)
            p = orig.subList(0, nummax);

        ArrayList<String> result = new ArrayList<>();
        for (int i = 0; i < p.size(); i++)
            result.add(p.get(i).venueId);
        return result;
    }

    /**
     * read the result from file
     * @param fName         file name of result file
     * @throws IOException
     */
    private void readResultFile(String fName) throws IOException {
        try(BufferedReader br = new BufferedReader(new FileReader(fName))) {
            String line = br.readLine(); // meta  info of model
            parseFirstLine(line);

            line = br.readLine(); // "users:"
            line = br.readLine();
            while(!line.equals("venues:")) {
                String[] comp = line.split(" ");
                String userId = comp[0];
                double[] factors = Utils.fromString(line.substring(userId.length() + 1));
                userMap.get(userId).setFactors(factors);
                line = br.readLine();
            }

            line = br.readLine();
            while(line != null) {
                String[] comp = line.split(" ");
                String vId = comp[0];
                double[] factors = Utils.fromString(line.substring(vId.length() + 1));
                venueMap.get(vId).setFactors(factors);
                line = br.readLine();
            }
        }
    }

    private void parseFirstLine(String line) {
        String[] comp = line.split(";");

        // parse k
        String[] kInfo = comp[0].split("=");
        double k_value = Integer.parseInt(kInfo[1]);
        assert (k == k_value);

        //parse isSigmoid
        String[] isSInfo = comp[1].split("=");
//        boolean isS_value = Boolean.parseBoolean(isSInfo[1]);
        double steepness_value;
        try {
            steepness_value = Double.parseDouble(isSInfo[1]);
        } catch (NumberFormatException ex) {
            // this one will handle the case that comp[1] is something like isSigmoid={true|false}.
            // This is the old format
            boolean isS_value = Boolean.parseBoolean(isSInfo[1]);
            if (isS_value)
                steepness_value = 1.0; // Sigmoid
            else
                steepness_value = 2.0; // tanh
        }
        assert (steepness == steepness_value);
//        assert (isS_value == isSigmoid);

        // parse isFriend
        String[] isFInfo = comp[2].split("=");
        boolean isF_value = Boolean.parseBoolean(isFInfo[1]);
        assert (isF_value == isFriend);

        // parse lambda
        double lu = Double.parseDouble(comp[3].split("=")[1]);
        double lv = Double.parseDouble(comp[4].split("=")[1]);
        double lf = Double.parseDouble(comp[5].split("=")[1]);
        this.params = new Parameters(lu, lv, lf);
    }

    /**
     * Read the ground truth file
     * @param fname         file contains the test set
     * @return              map of ground truth
     * @throws IOException
     */
    private HashMap<String, HashMap<String, Double>> readGroundTruth(String fname) throws IOException {
        HashMap<String, HashMap<String, Double>> result = new HashMap<>();

        try(BufferedReader br = new BufferedReader(new FileReader(fname))) {
            String line = br.readLine();

            while (line != null) {
                String[] comp = line.split(",");

                double numCks = Double.parseDouble(comp[0]);
                String userId = comp[1];
                String vId = comp[2];

                HashMap<String, Double> v = result.get(userId);
                if (v == null) {
                    v = new HashMap<>();
                    result.put(userId, v);
                }
                v.put(vId, numCks);

                line = br.readLine();
            }
        }
        return result;
    }
}

class PairObject {
    String venueId;
    double cks;

    public PairObject(String venueId, double cks) {
        this.venueId = venueId;
        this.cks = cks;
    }
}