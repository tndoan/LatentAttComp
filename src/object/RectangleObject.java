package object;

import utils.Distance;

public class RectangleObject {
	/**
	 * construction of object
	 * @param northeast far most point in the top right
	 * @param southwest far most point in the bottom left
	 */
	public RectangleObject(PointObject northeast, PointObject southwest) {
		super();
		this.northeast = northeast;
		this.southwest = southwest;
		this.center = new PointObject((northeast.getLat() + southwest.getLat())/2.0, (northeast.getLng() + southwest.getLng())/2.0);
	}

	/**
	 * far most point in the top right
	 */
	private PointObject northeast;
	
	/**
	 * far most point in the bottom right
	 */
	private PointObject southwest;
	
	/**
	 * center of the rectangle
	 */
	private PointObject center;

	/**
	 * getter of northeast point
	 * @return
	 */
	public PointObject getNortheast() {
		return northeast;
	}

	/**
	 * getter of southwest point
	 * @return
	 */
	public PointObject getSouthwest() {
		return southwest;
	}
	
	/**
	 * detect if the point is in rectangle
	 * @param p the point which we want to verify
	 * @return true if p is in rectangle; false otherwise 
	 */
	public boolean isInRectangle(PointObject p){
		if (p.getLat() >= this.southwest.getLat() & p.getLng() >= this.southwest.getLng() // greater than lower point
				& p.getLat() < this.northeast.getLat() & p.getLng() < this.northeast.getLng()) // smaller than higher point
			return true;
		return false;
	}
	
	/**
	 * calculate the area of rectangle (square meter)
	 * @return the area of rectangle
	 */
	public double getArea(){
		PointObject p = new PointObject(this.southwest.getLat(), this.northeast.getLng());
		double x = Distance.calculateDistance(p, this.northeast);
		double y = Distance.calculateDistance(p, this.southwest);
		return x * y;
	}
	
	/**
	 * check if the rectangle is square
	 * @return true if it is square; otherwise, return false
	 */
	public boolean isSquare(){
		PointObject nw = new PointObject(southwest.getLat(), northeast.getLng());
		double d1 = Distance.calculateDistance(northeast, nw);
		double d2 = Distance.calculateDistance(nw, southwest);
		return d1 == d2;
	}
	
	public String toString(){
		return "SW:" + southwest.toString() + ";NE:" + northeast.toString();
	}

	/**
	 * getter of center point
	 * @return the center point of rectangle
	 */
	public PointObject getCenter() {
		return center;
	}
}
