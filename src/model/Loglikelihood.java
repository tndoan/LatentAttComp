package model;

import java.util.HashMap;
import java.util.Set;

import object.AreaObject;
import object.UserObject;
import object.VenueObject;
import utils.Function;

public class Loglikelihood {
	
	/**
	 * calculate the log likelihood of model
	 * @param userMap		map of user id and the actual object
	 * @param venueMap		map of venue id and the actual object
	 * @param areaMap		map of area id and the actual object
	 * @param isSigmoid		sigmoid function or not
	 * @param k				number of latent feature
	 * @return				log likelihood of model
	 */
	public static double calculateLLH(HashMap<String, UserObject> userMap, HashMap<String, VenueObject> venueMap, 
			HashMap<String, AreaObject> areaMap, boolean isSigmoid, int k){
		double llh = 0.0;
		
		HashMap<String, double[]> areaFactorCache = new HashMap<>();
		// user chooses area
		for (String userId : userMap.keySet()) {
			UserObject uo = userMap.get(userId);
			double[] uFactor = uo.getFactors();
			
			for (String venueId : venueMap.keySet()) {
				VenueObject vo = venueMap.get(venueId);
				double w = uo.retrieveNumCks(venueId);
				
				String areaId = vo.getAreaId();
				double[] aFactors = areaFactorCache.get(areaId);
				if (aFactors == null) {
					AreaObject ao = areaMap.get(areaId);
					Set<String> lOfVIds = ao.getSetOfVenueIds();
					aFactors = new double[k];

					for (String vId : lOfVIds) {
						VenueObject vOfAreaId = venueMap.get(vId);
						aFactors = Function.plus(aFactors, vOfAreaId.getFactors());
					}

					areaFactorCache.put(areaId, aFactors);
				}

				llh += w * Math.log(Function.innerProduct(uFactor, aFactors));
			}
		}

		// venue win over their neighbors
		for (String userId : userMap.keySet()) {
			UserObject uo = userMap.get(userId);
			double[] uFactor = uo.getFactors();
			Set<String> lOfVenues = uo.getAllVenues();
			
			for (String vId : lOfVenues) {
				VenueObject vo = venueMap.get(vId);
				double w = uo.retrieveNumCks(vId);

				Set<String> neighbors = vo.getNeighbors();
				double lhs = Function.innerProduct(uFactor, vo.getFactors());

				double subLLH = 0.0;
				for (String nId : neighbors) {
					VenueObject no = venueMap.get(nId);
					double rhs = Function.innerProduct(uFactor, no.getFactors());

					double diff = lhs - rhs;
					if (isSigmoid)
						diff = Math.log(Function.sigmoidFunction(diff));
					else
						diff = Math.log(Function.tanh1_2(diff));
					subLLH += diff;
				}
				
				llh += w * subLLH;
			}
		}
		
		return llh;
	}
}
