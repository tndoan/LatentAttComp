package model;

import java.io.IOException;
import java.util.*;

import object.AreaObject;
import object.PointObject;
import object.UserObject;
import object.VenueObject;
import utils.Function;
import utils.ReadFile;
import utils.Utils;

public class Model {
	
	/**
	 * this is used to indicate the function of winning of venue over its neighbors
	 * if true, it is modeled as sigmoid function
	 * else, use CDF function
	 */
	private boolean isSigmoid;
	
	/**
	 * key is venue id, value is venue object corresponding to the venue id
	 */
	private HashMap<String, VenueObject> venueMap;

	/**
	 *
	 */
	private boolean isFriend;
	
	
	/**
	 * key is user id, value is user object corresponding to user id
	 */
	private HashMap<String, UserObject> userMap;

	/**
	 * store regularizer values
	 */
	private Parameters params;
	
	private int k;
	
	/**
	 * key is the area id(same as venue id), value is area object 
	 */
	private HashMap<String, AreaObject> areaMap;

	public Model(String uFile, String venueLocFile, String cksFile, boolean isSigmoid, int k, double scale, boolean isFriend) {
		this(uFile, venueLocFile, cksFile, isSigmoid, k, scale, isFriend,
				new Parameters(0.01, 0.01, 0.01));
	}

	public Model(String uFile, String venueLocFile, String cksFile, boolean isSigmoid, int k, double scale,
				 boolean isFriend, Parameters params) {
		this.isFriend = isFriend;
		this.params = params;
		this.isSigmoid = isSigmoid;
		this.k = k;
		
		// initialize 
		venueMap = new HashMap<>();
		userMap = new HashMap<>();
		

		//read data from files
		HashMap<String, String> vInfo = ReadFile.readLocation(venueLocFile);
		HashMap<String, HashMap<String, Integer>> cksMap = ReadFile.readNumCksFile(cksFile);

		HashMap<String, ArrayList<String>> userOfVenueMap = Utils.collectUsers(cksMap);

		// make venue object
		HashMap<String, PointObject> vLocInfo = new HashMap<>();
		for (String vId : vInfo.keySet()) {
			PointObject p = new PointObject(vInfo.get(vId));
			vLocInfo.put(vId, p);
		}
		
		HashMap<String, Integer> countMap = Utils.countCks(cksMap);

		areaMap = new HashMap<>();
		venueMap = Utils.createNeighborsBox(vLocInfo, areaMap, countMap, userOfVenueMap, scale, k);

		// make user object
		Set<String> uSet = cksMap.keySet();
		for (String uId : uSet) {
			HashMap<String, Integer> checkinMap = cksMap.get(uId);
			UserObject u = new UserObject(uId, checkinMap, k);
			userMap.put(uId, u);
		}

		System.out.println("# of users:" + userMap.keySet().size());
		System.out.println("# of venues:" + venueMap.keySet().size());
		System.out.println("# of areas:" + areaMap.keySet().size());
		System.gc();
	}

	/**
	 * Learning latent factors of users and venues inside the model via stochastic gradient descent
	 */
	public void learnParameters() {
		Set<String> allUIds = userMap.keySet();
		Set<String> allVIds = venueMap.keySet();
		boolean conv = false;
		long sTime = System.currentTimeMillis();
		double prevLLH = calculateParallelLLH();
		double learningRate = -0.000001;

		System.out.println(prevLLH + " in " + (System.currentTimeMillis() - sTime)/1000 + " s");
		while(!conv) {

			sTime = System.currentTimeMillis();
			Map<String, double[]> uGradMap = Collections.synchronizedMap(new HashMap<>());
			// calculate gradient of users
			allUIds.parallelStream().forEach(uId -> {
				double[] uGrad = userGrad(uId);
				uGradMap.put(uId, uGrad);
			});

			// update factor of users
			allUIds.parallelStream().forEach(uId -> {
				UserObject uo = userMap.get(uId);
				double[] uGrad = uGradMap.get(uId);
				double[] newUGrad = Function.minus(uo.getFactors(), Function.multiply(learningRate, uGrad));
				uo.setFactors(newUGrad);
			});
			System.out.println("sub uLLH:" + calculateLLH() + " in " + (System.currentTimeMillis() - sTime)/1000 + "s");

			sTime = System.currentTimeMillis();
			// calculate gradient of venues
			Map<String, double[]> vGradMap = Collections.synchronizedMap(new HashMap<>());
			allVIds.parallelStream().forEach(vId ->{
				double[] vGrad = venueGrad(vId);
				vGradMap.put(vId, vGrad);
			});

			// update factor of venues
			allVIds.parallelStream().forEach(vId -> {
				VenueObject vo = venueMap.get(vId);
				double[] vGrad = vGradMap.get(vId);
				double[] newVGrad = Function.minus(vo.getFactors(), Function.multiply(learningRate, vGrad));
				vo.setFactors(newVGrad);
			});

			double llh = calculateParallelLLH();
			System.out.println(llh + " in " + (System.currentTimeMillis() - sTime)/1000 + " s");
			if (Math.abs((llh - prevLLH) / prevLLH) < 0.01)
				conv = true;
			else
				prevLLH = llh;
		}
	}

	/**
	 * In this method, each user-venue pair is considered as one data point. Each time, one pair is given to do
	 * Stochastic gradient descend
	 */
	public void learnParametersStochastic() {
		ArrayList<String> allUIds = new ArrayList<>(userMap.keySet());
		boolean conv = false;
		long sTime = System.currentTimeMillis();
		double prevLLH = calculateParallelLLH();
		double learningRate = -0.000001;

		System.out.println(prevLLH + " in " + (System.currentTimeMillis() - sTime)/1000 + " s");
		while(!conv) {
			sTime = System.currentTimeMillis();

			for(String uId: allUIds) {
				UserObject uo = userMap.get(uId);
				for (String vId : uo.getAllVenues()){
					double[] uGrad = userGrad(uId, vId);
					uo.setFactors(Function.minus(uo.getFactors(), Function.multiply(learningRate, uGrad)));

					VenueObject vo = venueMap.get(vId);
					double[] vGrad = venueGrad(uId, vId);
					vo.setFactors(Function.minus(vo.getFactors(), Function.multiply(learningRate, vGrad)));
				}
			}

			double llh = calculateParallelLLH();
			System.out.println(llh + " in " + (System.currentTimeMillis() - sTime)/1000 + " s");
			if (Math.abs((llh - prevLLH) / prevLLH) < 0.01)
				conv = true;
			else
				prevLLH = llh;
		}
	}
	
	/**
	 * calculate the gradient of user
	 * @param userId	id of user
	 * @return			gradient vector of latent factor
	 */
	private double[] userGrad(String userId) {
		double[] grad = new double[k];
		UserObject uo = userMap.get(userId);
		double[] uFactor = uo.getFactors();
		
		// 1st part
		Set<String> lOfVenues = uo.getAllVenues();
		for (String vId : lOfVenues) {
			VenueObject vo = venueMap.get(vId);
			String aId = vo.getAreaId();
			AreaObject ao = areaMap.get(aId);
			double[] aFactor = new double[k];
			
			for (String venueId : ao.getSetOfVenueIds()) {
				VenueObject venueObj = venueMap.get(venueId);
				aFactor = Function.plus(aFactor, venueObj.getFactors());
			}
			
			double w = uo.retrieveNumCks(vId);
			double denominator = w / Function.innerProduct(uFactor, aFactor);
			
			grad = Function.plus(grad, Function.multiply(denominator, aFactor));
		}
		
		// 2nd part
		for (String vId : lOfVenues) {
			VenueObject vo = venueMap.get(vId);
			Set<String> neighbors = vo.getNeighbors();
			double lhs = Function.innerProduct(uFactor, vo.getFactors());
			
			double[] sub = new double[k];
			for (String nId : neighbors) {
				VenueObject neighborObj = venueMap.get(nId);
				double rhs = Function.innerProduct(uFactor, neighborObj.getFactors());
				double diff = lhs - rhs;
				double p;
				double[] subVector;
				
				if (isSigmoid) {
					p = - Function.sigmoidFunction(diff) * Math.exp(- diff);
					subVector = Function.minus(neighborObj.getFactors(), vo.getFactors());
					subVector = Function.multiply(p, subVector);
				} else {
					p = 1.0/ Function.tanh1_2(diff);
					double d = 1.0 + Math.exp(-2.0 * diff);
					p *= 2.0 * Math.exp(-2.0 * diff) / (d * d);
					subVector = Function.minus(vo.getFactors(), neighborObj.getFactors());
					subVector = Function.multiply(p, subVector);
				}
				
				sub = Function.plus(sub, subVector);
			}
			sub = Function.multiply(uo.retrieveNumCks(vId), sub);
			grad = Function.plus(sub, grad);
		}

		// regularization
		grad = Function.plus(grad, Function.multiply(-2.0 * params.getLambda_u(), uFactor));

		return grad;
	}
	
	/**
	 * calculate gradient vector of latent feature of venue
	 * @param venueId	id of venue
	 * @return			gradient vector of latent factor of venue
	 */
	private double[] venueGrad(String venueId) {
		double[] grad = new double[k];
		VenueObject vo = venueMap.get(venueId);
		double[] vFactor = vo.getFactors();
		String aId = vo.getAreaId();
		AreaObject ao = areaMap.get(aId);
		Set<String> setOfVenues = ao.getSetOfVenueIds();
		
		// 1st part
		for (String vId : setOfVenues) {
			VenueObject venueObj = venueMap.get(vId);
			ArrayList<String> userIds = venueObj.getUserIds();
			for (String uId : userIds) {
				UserObject uo = userMap.get(uId);
				double[] uFactor = uo.getFactors();
				
				double[] sub = new double[k];
				for (String vPrime : setOfVenues) 
					sub = Function.plus(sub, venueMap.get(vPrime).getFactors());
				
				double argument = uo.retrieveNumCks(vId) / Function.innerProduct(uFactor, sub);
				double[] comp = Function.multiply(argument, uFactor);
				grad = Function.plus(comp, grad);
			}
		}
		
		// 2nd part
		ArrayList<String> uList = vo.getUserIds();
		Set<String> neighborIds = vo.getNeighbors();
		
		for (String uId : uList) {
			UserObject uo = userMap.get(uId);
			double[] uFactor = uo.getFactors();
			double lhs = Function.innerProduct(uFactor, vFactor);
			
			double[] sub = new double[k];
			for (String nId : neighborIds) {
				VenueObject nObj = venueMap.get(nId);
				double rhs = Function.innerProduct(uFactor, nObj.getFactors());
				double diff = lhs - rhs;
				double p;
				
				if (isSigmoid)
					p = Math.exp(-diff) * Function.sigmoidFunction(diff);
				else {
					double e = Math.exp(-2.0 * diff);
					p = 2.0 * e / (Function.tanh1_2(diff) * (e + 1.0) * (1.0 + e));
				}
				
				sub = Function.plus(sub, Function.multiply(p, uFactor));
			}
			
			sub = Function.multiply(uo.retrieveNumCks(venueId), sub);
			grad = Function.plus(sub, grad);
		}
		
		// 3rd part
		for (String nId : neighborIds) {
			VenueObject nObj = venueMap.get(nId);
			ArrayList<String> userList = nObj.getUserIds();
			
			for (String uId : userList) {
				UserObject uo = userMap.get(uId);
				double[] uFactor = uo.getFactors();
				double lhs = Function.innerProduct(uFactor, nObj.getFactors());
				double rhs = Function.innerProduct(uFactor, vFactor);
				double diff = lhs - rhs;
				double p;
				
				if (isSigmoid)
					p = - Function.sigmoidFunction(diff) * Math.exp(-diff);
				else {
					double e = Math.exp(-2.0 * diff);
					p = - 2.0 * e / ((1.0 + e) * (1.0 + e) * Function.tanh1_2(diff));
				}
				
				double[] sub = Function.multiply(p * uo.retrieveNumCks(nId), uFactor);
				grad = Function.plus(sub, grad);
			}
		}

		// regularization
		grad = Function.plus(grad, Function.multiply(-2.0 * params.getLambda_v(), vFactor));

		return grad;
	}

	/**
	 * 
	 * @return	calculate the log likelihood of model
	 */
	public double calculateLLH() {
		return Loglikelihood.calculateLLH(userMap, venueMap, areaMap, isSigmoid, k, params);
	}

	/**
	 * calculate log likelihood with highly parallel
	 * @return	log likelihood
	 */
	public double calculateParallelLLH() {
		return Loglikelihood.calculateParallelLLH(userMap, venueMap, areaMap, isSigmoid, k, params);
	}

	private double[] userGrad(String uId, String vId) {
		UserObject u = userMap.get(uId);
		double[] uFactor = u.getFactors();
		VenueObject v = venueMap.get(vId);
		double[] vFactor = v.getFactors();
		AreaObject ao = areaMap.get(v.getAreaId());
		double[] aFactor = new double[k];
		for (String venueId : ao.getSetOfVenueIds()) {
			double[] venueFactor = venueMap.get(venueId).getFactors();
			aFactor = Function.plus(aFactor, venueFactor);
		}
		double denominator = Function.innerProduct(u.getFactors(), aFactor);

		double[] result = Function.multiply(1.0/ denominator, aFactor);

		double lhs = Function.innerProduct(v.getFactors(), uFactor);
		double[] l2 = new double[k];

		for(String nId : v.getNeighbors()) {
			VenueObject nObj = venueMap.get(nId);
			double rhs = Function.innerProduct(nObj.getFactors(), uFactor);
			double diff = lhs - rhs;
			double[] diffVector = Function.minus(nObj.getFactors(), vFactor);
			double inFront;
			if (isSigmoid)
				inFront = -Function.sigmoidFunction(diff) * Math.exp(-diff);
			else
				inFront = - 2.0 * Math.exp(-2.0 * diff) / (1.0 + Math.exp(-2.0 * diff));

			l2 = Function.plus(l2, Function.multiply(inFront, diffVector));
		}
		result = Function.plus(l2, result);

		double[] r = Function.multiply(-2.0 * params.getLambda_u(), uFactor);

		return Function.plus(Function.multiply(u.retrieveNumCks(vId), result), r);
	}

	private double[] venueGrad(String uId, String vId) {
		UserObject uObj = userMap.get(uId);
		double[] uFactor = uObj.getFactors();

		VenueObject vObj = venueMap.get(vId);
		double[] vFactor = vObj.getFactors();
		Set<String> neighborIds = vObj.getNeighbors();
		AreaObject aObj = areaMap.get(vObj.getAreaId());

		double[] aFactor = new double[k];
		for (String nId : aObj.getSetOfVenueIds()){
			VenueObject nObj = venueMap.get(nId);
			aFactor = Function.plus(aFactor, nObj.getFactors());
		}
		double d = Function.innerProduct(uFactor, aFactor);
		double[] result = Function.multiply(1.0 / d, uFactor);

		double lhs = Function.innerProduct(uFactor, vFactor);
		double total = 0.0;
		for (String nId : neighborIds) {
			VenueObject nObj = venueMap.get(nId);
			double rhs = Function.innerProduct(uFactor, nObj.getFactors());
			double diff = lhs - rhs;

			if (isSigmoid) {
				double eDiff = Math.exp(-diff);
				double multiplier = eDiff / (1.0 + eDiff);
				total += multiplier;
			} else {
				double e2Diff = Math.exp(-2.0 * diff);
				double multiplier = 2.0 * e2Diff / (1.0 + e2Diff);
				total += multiplier;
			}
		}
		result = Function.plus(result, Function.multiply(total, uFactor));

		// regularization
		double[] r = Function.multiply(-2.0 * params.getLambda_v(), vFactor);

		return Function.plus(Function.multiply(uObj.retrieveNumCks(vId), result), r);
	}

	public double calculateLLH(String uId, String vId) {
		return Loglikelihood.calculateLLH(uId, vId, userMap, venueMap, areaMap, isSigmoid, k, params);
	}

	public void writeModel(String prefix) throws IOException {

		ArrayList<String> result = new ArrayList<>();
		// parameters
		String parameters = "k=" + k + ";isSigmoid=" + isSigmoid + ";isFriend=" + isFriend
				+ ";lambda_u=" + params.getLambda_u() + ";lambda_v=" + params.getLambda_v() + ";lambda_f="
				+ params.getLambda_f();
		result.add(parameters);

		// user
		result.add("users:");
		for (String uId : userMap.keySet()) {
			StringBuffer sb = new StringBuffer();
			sb.append(uId + " ");
			UserObject uo = userMap.get(uId);
			sb.append(Arrays.toString(uo.getFactors()));
			result.add(sb.toString());
		}

		// venue
		result.add("venues:");
		for (String vId : venueMap.keySet()) {
			StringBuffer sb = new StringBuffer();
			sb.append(vId + " ");
			VenueObject vo = venueMap.get(vId);
			sb.append(Arrays.toString(vo.getFactors()));
			result.add(sb.toString());
		}
	}

	/**
	 * it is used for testing gradient only so it is set to be private
	 * @param uId	user id
	 * @return		user object
	 */
	private UserObject getUO(String uId) {
		return userMap.get(uId);
	}

	/**
	 * it is used for testing gradient only so it is set to be private
	 * @param vId	venue id
	 * @return		venue object
	 */
	private VenueObject getVO(String vId) {
		return venueMap.get(vId);
	}

	public static void main(String[] args){
        Model m = new Model("data/uLoc.txt",
                "data/vLoc.txt",
                "data/cks.txt",
                true,
                5,
                0.05,
				false);

		String uId = "1"; String vId = "1";

//		UserObject o = m.getUO(uId);
		VenueObject o = m.getVO(vId);
		o.setFactors(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
		double eps = 0.0001;
//		double[] grad = m.userGrad(uId, vId);
		double[] grad = m.venueGrad(uId, vId);
		System.out.println(grad[0]);

		System.out.println("----");
		o.setFactors(new double[]{1.0 + eps, 2.0, 3.0, 4.0, 5.0});
		double llh1 = m.calculateLLH(uId, vId);
		o.setFactors(new double[]{1.0 - eps, 2.0, 3.0, 4.0, 5.0});
		double llh2 = m.calculateLLH(uId, vId);
		System.out.println((llh1 - llh2) /(2.0 * eps));
	}
}
