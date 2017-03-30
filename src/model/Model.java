package model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

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
	 * key is user id, value is user object corresponding to user id
	 */
	private HashMap<String, UserObject> userMap;
	
	private int k;
	
	/**
	 * key is the area id(same as venue id), value is area object 
	 */
	private HashMap<String, AreaObject> areaMap;

	public Model(String uFile, String venueLocFile, String cksFile, double scope, boolean isSigmoid, int k, double scale, 
			boolean isAverageLocation) {
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
		venueMap = Utils.createNeighborsBox(vLocInfo, areaMap, countMap, userOfVenueMap, scale, isAverageLocation, k);

		// make user object
		Set<String> uSet = cksMap.keySet();
		for (String uId : uSet) {
			HashMap<String, Integer> checkinMap = cksMap.get(uId);
			UserObject u = new UserObject(uId, checkinMap, k);
			userMap.put(uId, u);
		}
	}

	/**
	 * Learning latent factors of users and venues inside the model
	 */
	public void learnParameters() {
		boolean conv = false;
		double prevLLH = calculateLLH();
		double learningRate = 0.01;

		while(!conv) {
			// update factor of users
			for (String uId : userMap.keySet()) {
				UserObject uo = userMap.get(uId);
				double[] uGrad = userGrad(uId);
				double[] newUGrad = Function.minus(uGrad, Function.multiply(learningRate, uGrad));
				uo.setFactors(newUGrad);
			}
			
			// update factor of venues
			for (String vId : venueMap.keySet()) {
				VenueObject vo = venueMap.get(vId);
				double[] vGrad = venueGrad(vId);
				double[] newVGrad = Function.minus(vGrad, Function.multiply(learningRate, vGrad));
				vo.setFactors(newVGrad);
			}
			
			double llh = calculateLLH();
			if (Math.abs(llh - prevLLH) < 0.001)
				conv = true;
			else
				prevLLH = llh;
		}
	}
	
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
			
			grad = Function.plus(aFactor, Function.multiply(denominator, aFactor));
		}
		
		// 2nd part
		for (String vId : lOfVenues) {
			VenueObject vo = venueMap.get(vId);
			ArrayList<String> neighbors = vo.getNeighbors();
			double lhs = Function.innerProduct(uFactor, vo.getFactors());
			
			double[] sub = new double[k];
			for (String nId : neighbors) {
				VenueObject neighborObj = venueMap.get(nId);
				double rhs = Function.innerProduct(uFactor, neighborObj.getFactors());
				double diff = lhs - rhs;
				double p = 1.0;
				double[] subVector = new double[k];
				
				if (isSigmoid) {
					p = - Function.sigmoidFunction(diff) * Math.exp(- diff);
					subVector = Function.minus(neighborObj.getFactors(), vo.getFactors());
					subVector = Function.multiply(p, subVector);
				} else {
					p = 1.0/ Function.cdf(diff);
					p *= Function.normal(diff);
					subVector = Function.minus(vo.getFactors(), neighborObj.getFactors());
					subVector = Function.multiply(p, subVector);
				}
				
				sub = Function.plus(sub, subVector);
			}
			sub = Function.multiply(uo.retrieveNumCks(vId), sub);
			grad = Function.plus(sub, grad);
		}

		return grad;
	}
	
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
		ArrayList<String> neighborIds = vo.getNeighbors();
		
		for (String uId : uList) {
			UserObject uo = userMap.get(uId);
			double[] uFactor = uo.getFactors();
			double lhs = Function.innerProduct(uFactor, vFactor);
			
			double[] sub = new double[k];
			for (String nId : neighborIds) {
				VenueObject nObj = venueMap.get(nId);
				double rhs = Function.innerProduct(uFactor, nObj.getFactors());
				double diff = lhs - rhs;
				double p = 1.0;
				
				if (isSigmoid)
					p = Math.exp(-diff) * Function.sigmoidFunction(diff);
				else 
					p = Function.normal(diff) / Function.cdf(diff);
				
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
				double p = 1.0;
				
				if (isSigmoid)
					p = - Function.sigmoidFunction(diff) * Math.exp(-diff);
				else
					p = - Function.normal(diff) / Function.cdf(diff);
				
				double[] sub = Function.multiply(p * uo.retrieveNumCks(nId), uFactor);
				grad = Function.plus(sub, grad);
			}
		}
		
		return grad;
	}

	/**
	 * 
	 * @return	calculate the log likelihood of model
	 */
	private double calculateLLH() {
		return Loglikelihood.calculateLLH(userMap, venueMap, areaMap, isSigmoid, k);
	}
	
	public static void main(String[] args) {
		System.out.println("print ");
	}
}
