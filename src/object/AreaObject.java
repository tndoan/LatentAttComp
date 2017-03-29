package object;

import java.util.Set;

public class AreaObject {
	
	private PointObject location;
	
	private String id;
	
	private Set<String> setOfVenueIds;
	
	/**
	 * construction for area object
	 * @param id
	 * @param location
	 * @param setOfVenueIds
	 */
	public AreaObject(String id, PointObject location, Set<String> setOfVenueIds){
		this.id = id;
		this.location = location;
		this.setOfVenueIds = setOfVenueIds;
	}

	public PointObject getLocation() {
		return location;
	}

	public String getId() {
		return id;
	}

	public Set<String> getSetOfVenueIds() {
		return setOfVenueIds;
	}
	
	/**
	 * print out info of this area
	 */
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append("id:" + id + "\n");
		sb.append("set of venues:" );
		for (String venue : setOfVenueIds) {
			sb.append(venue + ",");
		}
		sb.append("\n");
		sb.append("location:" + location.toString() + "\n");
		return sb.toString();
	}
}
