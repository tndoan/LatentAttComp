package utils;

import object.PointObject;

/**
 * 
 * @author tndoan
 *
 */
public class Distance {
	
	public final static double AVERAGE_RADIUS_OF_EARTH = 6371.0;
	
	/**
	 * 
	 * @param p1
	 * @param p2
	 * @return		Square of Euclidean distance between 2 points 
	 */
	public static double calSqEuDistance(PointObject p1, PointObject p2) {
		double x = p1.getLat() - p2.getLat();
		double y = p1.getLng() - p2.getLng();
		
		return x*x + y*y;
	}
	
	/**
	 * 
	 * @param p1
	 * @param p2
	 * @return		distance in meter between 2 points
	 */
	public static double calculateDistance(PointObject p1, PointObject p2){
	    double latDistance = Math.toRadians(p1.getLat() - p2.getLat());
	    double lngDistance = Math.toRadians(p1.getLng() - p2.getLng());

	    double a = (Math.sin(latDistance / 2) * Math.sin(latDistance / 2)) +
	                    (Math.cos(Math.toRadians(p1.getLat()))) *
	                    (Math.cos(Math.toRadians(p2.getLat()))) *
	                    (Math.sin(lngDistance / 2)) *
	                    (Math.sin(lngDistance / 2));

	    double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

	    return (double) AVERAGE_RADIUS_OF_EARTH * c * 1000.0;   
	}
}
