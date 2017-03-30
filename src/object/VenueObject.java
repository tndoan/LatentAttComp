package object;

import java.util.ArrayList;
import java.util.Random;

/**
 * 
 * @author tndoan
 *
 */
public class VenueObject {
	
	public VenueObject(String id, int totalCks,	PointObject location, ArrayList<String> neighbors, ArrayList<String> userIds, int k){
		this.id = id;
		this.location = location;
		this.neighbors = neighbors;
		this.userIds = userIds;
		this.totalCks = totalCks;
		this.isSetArea = false;
		
		Random generator = new Random();
		this.factors = new double[k];
		for (int i = 0; i < k; i++)
			this.factors[i] = 5.0 * generator.nextDouble() + 2.0;
	}
	
	/**
	 * latent factor vector
	 */
	private double[] factors;
	
	public double[] getFactors() {
		return factors;
	}

	public void setFactors(double[] factors) {
		this.factors = factors;
	}
	
	private boolean isSetArea;
	
	/**
	 * indicate the area id that this venue is belong to
	 */
	private String areaId;
	
	/**
	 * total number of check-in that it has
	 */
	private int totalCks;
	
	/**
	 * location of venue
	 */
	private PointObject location;
	
	/**
	 * id of venue
	 */
	private String id;
	
	/**
	 * list of id of neighbors
	 */
	private ArrayList<String> neighbors;
	
	/**
	 * list of user ids who have check-in in this venue
	 */
	private ArrayList<String> userIds;

	public PointObject getLocation() {
		return location;
	}

	public String getId() {
		return id;
	}

	public ArrayList<String> getNeighbors() {
		return neighbors;
	}

	public ArrayList<String> getUserIds() {
		return userIds;
	}

	public int getTotalCks() {
		return totalCks;
	}

	public String getAreaId() {
		return areaId;
	}

	/**
	 * can set area id for venue one time. The second time will be ignored.
	 * @param areaId
	 */
	public void setAreaId(String areaId) {
		// if not set area, set it
		// otherwise, ignore
		if (this.isSetArea == false) {
			this.areaId = areaId;
			this.isSetArea = true;
		}
	}
	
}
