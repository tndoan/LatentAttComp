package utils;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import object.AreaObject;
import object.PointObject;
import object.RectangleObject;
import object.VenueObject;

public class Utils {
	/**
	 * 
	 * @param cksMap
	 * @return
	 */
	public static HashMap<String, Integer> countCks(HashMap<String, HashMap<String, Integer>> cksMap){
		HashMap<String, Integer> result = new HashMap<>();
		
		for (String userId : cksMap.keySet()){
			HashMap<String, Integer> map = cksMap.get(userId);
			for (String venueId : map.keySet()){
				int numCks = map.get(venueId);
				Integer currentCks = result.get(venueId);
				if (currentCks == null){
					currentCks = 0;
				} 
				result.put(venueId, numCks + currentCks);
			}
		}
		
		return result;
	}

	/**
	 * 
	 * @param cksMap
	 * @return hash map whose key is venue id, value is list of user id who have check-in in this venue
	 */
	public static HashMap<String, ArrayList<String>> collectUsers(HashMap<String, HashMap<String, Integer>> cksMap){
		HashMap<String, ArrayList<String>> result = new HashMap<>();
		
		for (String userId : cksMap.keySet()){
			HashMap<String, Integer> map = cksMap.get(userId);
			
			for (String venueId : map.keySet()){
				Integer numCks = map.get(venueId);
				if (numCks == null){
					System.err.println("Bug here");
					continue;
				}
				
				ArrayList<String> listOfUId = result.get(venueId);
				if (listOfUId == null){
					listOfUId = new ArrayList<>();
					result.put(venueId, listOfUId);
				}
				
				listOfUId.add(userId);
			}
		}
		
		return result;
	}
	
	/**
	 * Round up number up to the places-th behind the point
	 * For example, 11.56 => 11.6
	 * @param value		value we want to round up
	 * @param places	location behind the point that we want to round up
	 * @return			round up value
	 */
	public static double roundUp(double value, int places) {
	    if (places < 0) throw new IllegalArgumentException();

	    double factor = Math.pow(10, places);
	    value = value * factor;
	    double tmp = Math.ceil(value);
	    return tmp / factor;
	}
	
	/**
	 * Round down number up to the places-th behind the point
	 * Eg: 11.56 => 11.5
	 * @param value		value we want to round down
	 * @param places	location behind the point that we want to round down
	 * @return
	 */
	public static double roundDown(double value, int places){
		if (places < 0) throw new IllegalArgumentException();

	    double factor = Math.pow(10, places);
	    value = value * factor;
	    double tmp = Math.floor(value);
	    return tmp / factor;
	}

	/**
	 * 
	 * @param vInfo
	 * @param countMap
	 * @param userOfVenueMap
	 * @param areaMap
	 * @param isAverageLocation
	 * @param threshold
	 * @return
	 */
	public static HashMap<String, VenueObject> createNeighborsList(HashMap<String, String> vInfo, HashMap<String, Integer> countMap,
			HashMap<String, ArrayList<String>> userOfVenueMap, HashMap<String, AreaObject> areaMap, boolean isAverageLocation, 
			double threshold, int k) {		
		// create neighbors map
		HashMap<String, ArrayList<String>> neighbors = new HashMap<>();
		
		Set<String> allVenueIds = vInfo.keySet();
		ArrayList<String> venueIdsList = new ArrayList<>();
		venueIdsList.addAll(allVenueIds);
		
		for (int i = 0; i < venueIdsList.size(); i++) {
			String v1 = venueIdsList.get(i);
			PointObject p1 = new PointObject(vInfo.get(v1)); // location of point i
			for (int j = i + 1; j < venueIdsList.size(); j++) {
				String v2 = venueIdsList.get(j);
				PointObject p2 = new PointObject(vInfo.get(v2)); // location of point j
				
				double dist = Distance.calculateDistance(p1, p2);
				
				if (dist < threshold) {
					// if distance is less than threshold, these two points are neighbors.
					ArrayList<String> l1 = neighbors.get(v1);
					if (l1 == null) {
						l1 = new ArrayList<>();
						neighbors.put(v1, l1);
					}
					l1.add(v2);
					
					ArrayList<String> l2 = neighbors.get(v2);
					if(l2 == null) {
						l2 = new ArrayList<>();
						neighbors.put(v2, l2);
					}
					l2.add(v1);
				}
			}
		}
		// finish building neighbor map
		
		// build venue map
		HashMap<String, VenueObject> result = new HashMap<>();
		for (String vId : allVenueIds) {
			//parse the location of venues
			String locInfo = vInfo.get(vId);
			PointObject location = new PointObject(locInfo);
			
			ArrayList<String> neighborIds = neighbors.get(vId);
			
			int numCks = countMap.get(vId);
			
			ArrayList<String> listOfUsers = userOfVenueMap.get(vId);
			
			VenueObject vo = new VenueObject(vId, numCks, location, neighborIds, listOfUsers, k);
			result.put(vId, vo);
		}
		
		// make area map
		areaMap = MakeAreaMap.createEachPointCluster(result, isAverageLocation);
		
		return result;
	}
	
	/**
	 * 
	 * @param vInfo
	 * @param areaMap
	 * @param countMap
	 * @param userOfVenueMap
	 * @param scale				size of the cell in degree
	 * @param isAverageLoc		true -> location of area is the average locations of all venues in this area; false -> center of the square
	 * @return
	 */
	public static HashMap<String, VenueObject> createNeighborsBox(HashMap<String, PointObject> vInfo, HashMap<String, AreaObject> areaMap, 
			HashMap<String, Integer> countMap, HashMap<String, ArrayList<String>> userOfVenueMap, double scale, 
			boolean isAverageLoc, int k) {
		Collection<PointObject> locInfo = vInfo.values();
		
		// find venues inside area
		RectangleObject coverRectangle = MakeAreaMap.surroundingGrid1(locInfo);
		System.out.println("cover rectangle:" + coverRectangle.toString());
//		double scale = 0.1; // the size of each square is 0.1 x 0.1 (latitude and longitude)
		
		PointObject ne = coverRectangle.getNortheast();
		PointObject sw = coverRectangle.getSouthwest();
		
		double base_min_lat = sw.getLat();
		double base_min_lng = sw.getLng();

		int numLat = (int) Math.round(ne.getLat() / scale - sw.getLat() / scale);
		int numLng = (int) Math.round(ne.getLng() / scale - sw.getLng() / scale);
		
		// I know it is not a good way to handle this case but maybe it works
		// key is area id; value is set of venue id which is belong to this area
		HashMap<String, Set<String>> venuesInArea = new HashMap<>();
		
		for (String vId : vInfo.keySet()) {
			PointObject loc = vInfo.get(vId);
			
			// cell id of this venue
			int i = (int) Math.floor((loc.getLat() - base_min_lat) / scale);
			int j = (int) Math.floor((loc.getLng() - base_min_lng) / scale);
			
			// area id of venue. Each venue is belong to only 1 area.
			String areaIds = String.valueOf(i * numLng  + j );
//			System.out.println("venueId:" + vId + "\ti:" + i + "\tj:" + j + "\tareaId:" + areaIds);
			
			Set<String> listOfVenues = venuesInArea.get(areaIds);
			if (listOfVenues == null) {
				listOfVenues = new HashSet<>();
				venuesInArea.put(areaIds, listOfVenues);
			}
			listOfVenues.add(vId);
		}
		
		
		// neighbors of a venue in this case are not only venues in the same box (area) with this venue but also 
		// venues in surrounding boxes of box of this venue. For example, neighbors of venue in box 5 also contain
		// venue in boxes 1 to 9
		// | 1 | 2 | 3 |
		// | 4 | 5 | 6 |
		// | 7 | 8 | 9 |
		HashMap<String, ArrayList<String>> neighbors = new HashMap<>();
		HashMap<String, String> areaIdOfVenue = new HashMap<>();
		for (int i = 0; i < numLat; i++ ) {
			for (int j = 0; j < numLng; j++) {
				String areaId = String.valueOf(i * numLng + j);
								
				Set<String> setOfVenues = venuesInArea.get(areaId);
				if (setOfVenues != null) { // if there are no venues in this area, we dont care
					for (String vId : setOfVenues) {
						// assign area id to venue
						areaIdOfVenue.put(vId, areaId);
						
						ArrayList<String> subneighbors = new ArrayList<>(setOfVenues);
						subneighbors.remove(vId);
						neighbors.put(vId, subneighbors);
						
						// list of surrounding areas
						ArrayList<String> ns = Utils.getNeighborArea(i, j, numLat, numLng);
						for (String n : ns) {
							// add all venues in surrounding areas as neighbors of venue
							Set<String> nes = venuesInArea.get(n);
							if (nes != null)
								subneighbors.addAll(nes);
						}
					}
				}
			}
		}
		
		// create venue map
		HashMap<String, VenueObject> venueMap = new HashMap<>();
		for (String venueId: vInfo.keySet()){
			PointObject location = vInfo.get(venueId);
			
			ArrayList<String> neighborIds = neighbors.get(venueId);

			Integer numCks = countMap.get(venueId);
			if (numCks == null)
				numCks = 0;
			
			ArrayList<String> listOfUsers = userOfVenueMap.get(venueId);
			
			VenueObject vo = new VenueObject(venueId, numCks, location, neighborIds, listOfUsers, k);
			
			String areaId = areaIdOfVenue.get(venueId);
			
			vo.setAreaId(areaId);
			
			venueMap.put(venueId, vo);
		}
		
		// create area
		for (int i = 0; i < numLat; i++) {
			for (int j = 0; j < numLng; j++) {
				String areaId = String.valueOf(i * numLng + j);

				Set<String> allVenueIds = venuesInArea.get(areaId);
				if (allVenueIds == null) // if there is no venue in this area, ignore
					continue;
				
				double average_lat = 0.0;
				double average_lng = 0.0;
				
				for (String vId : allVenueIds) {
					VenueObject v = venueMap.get(vId);
					
					average_lat += v.getLocation().getLat();
					average_lng += v.getLocation().getLng();
				}

				average_lat /= (double) allVenueIds.size();
				average_lng /= (double) allVenueIds.size();
				
				// location of area
				PointObject aLoc = null;
				if (isAverageLoc) {
					aLoc = new PointObject(average_lat, average_lng);
				} else {
					PointObject sub_ne = new PointObject(base_min_lat + (scale * (double)(i + 1)), base_min_lng + (scale * (double) (j + 1)));
					PointObject sub_sw = new PointObject(base_min_lat + (scale * (double)(i)), base_min_lng + (scale * (double) (j)));
					RectangleObject rObj = new RectangleObject(sub_ne, sub_sw);
					aLoc = rObj.getCenter();
				}

				AreaObject area = new AreaObject(areaId, aLoc, allVenueIds);
				areaMap.put(areaId, area);
			}
		}
		
		return venueMap;
	} 
	
	public static ArrayList<String> getNeighborArea(int i, int j, int numLat, int numLng) {
		ArrayList<String> result = new ArrayList<>();
		
		if ( i - 1 >= 0)
			result.add(String.valueOf((i - 1) * numLat + j));
		
		if (j - 1 >= 0)
			result.add(String.valueOf(i * numLat + j - 1));
		
		if (i + 1 < numLat) 
			result.add(String.valueOf((i + 1) * numLat + j));
		
		if (j + 1 < numLng)
			result.add(String.valueOf(i * numLat + j + 1));
		
		if (i - 1 >= 0 && j + 1 < numLng)
			result.add(String.valueOf((i - 1) * numLat + j + 1));
		
		if (i - 1 > 0 && j - 1 >= 0)
			result.add(String.valueOf((i - 1) * numLat + j - 1));
		
		if (i + 1 < numLat && j - 1 >= 0)
			result.add(String.valueOf((i + 1) * numLat + j - 1));
		
		if (i + 1 < numLat && j + 1 < numLng)
			result.add(String.valueOf((i + 1) * numLat + j + 1));
		
		return result;
	}
	
	/**
	 * write list of string to file whose name is given
	 * @param results						list of string. Note: Each string does not contain "break line"(\n) character
	 * @param fname							name of the file
	 * @throws UnsupportedEncodingException
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static void writeFile(ArrayList<String> results, String fname)
			throws UnsupportedEncodingException, FileNotFoundException,
			IOException {
		try (Writer writer = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(fname), "utf-8"))) {
			for (String s : results) {
				writer.write(s + "\n");
			}
		}
	}
	
	/**
	 * calculate the center of mass from the check-in map of users
	 * @param checkinMap	map whose key is venue id and value is number of check-ins
	 * @param vLocInfo		map whose key is venue id and value is the point object of venue
	 * @return				point object (center of the mass)
	 */
	public static PointObject calculateCenterOfMass(HashMap<String, Integer> checkinMap, HashMap<String, PointObject> vLocInfo) {
		double lat = 0.0;
		double lng = 0.0;
		double totalCks = 0.0;
		
		for (String vId : checkinMap.keySet()) {
			PointObject vPoint = vLocInfo.get(vId);
			double numCks = (double) checkinMap.get(vId);
			
			lat += vPoint.getLat() * numCks;
			lng += vPoint.getLng() * numCks;
			totalCks += numCks;
		}
		
		return new PointObject(lat/totalCks, lng/totalCks);
	}
}
