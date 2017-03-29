package object;

/**
 * all of point object
 * @author tndoan
 *
 */
public class PointObject {
	/**
	 * latitude of point object
	 */
	private double lat;
	
	/**
	 * longitude of point object
	 */
	private double lng;
	
	/**
	 * constructor
	 * @param lat	latitude
	 * @param lng	longitude
	 */
	public PointObject(double lat, double lng){
		this.lat = lat;
		this.lng = lng;
	}
	
	/**
	 * constructor from string
	 * @param s
	 */
	public PointObject(String s) {
		if (s.equals("?")) {
			// if location is unknown, assign 
			this.lat = 10.804200;
			this.lng = 106.695736;
		} else {
			String[] comp = s.split(",");
			this.lat = Double.parseDouble(comp[0]);
			this.lng = Double.parseDouble(comp[1]);
		}
	}

	/**
	 * get latitude of point
	 * @return	latitude
	 */
	public double getLat() {
		return lat;
	}

	/**
	 * get longitude of point
	 * @return	longitude
	 */
	public double getLng() {
		return lng;
	}
	
	/**
	 * to String
	 */
	public String toString(){
		return "latitude:" + lat + "\t longitude:" + lng;
	}
	
}
