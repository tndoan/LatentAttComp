package object;

import java.util.Set;

public class AreaObject {

	private String id;
	
	private Set<String> setOfVenueIds;
	
	/**
	 * construction for area object
	 * @param id
	 * @param setOfVenueIds
	 */
	public AreaObject(String id, Set<String> setOfVenueIds){
		this.id = id;
		this.setOfVenueIds = setOfVenueIds;
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
		return sb.toString();
	}
}
