package utils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import object.AreaObject;
import object.PointObject;
import object.RectangleObject;
import object.VenueObject;

public class MakeAreaMap {
	
	/**
	 * 
	 * @param vMap
	 * @param isAverageLocation
	 * @return
	 */
	public static HashMap<String, AreaObject> createEachPointCluster(HashMap<String, VenueObject> vMap, boolean isAverageLocation) {
		HashMap<String, AreaObject> result = new HashMap<>();
		
		for (String venueId : vMap.keySet()) {
			VenueObject vo = vMap.get(venueId);
			Set<String> venueIds = vo.getNeighbors();
			
			double lat = vo.getLocation().getLat();
			double lng = vo.getLocation().getLng();
			
			if (isAverageLocation) {
				for (String vId : venueIds){
					VenueObject v = vMap.get(vId);
					lat += v.getLocation().getLat();
					lng += v.getLocation().getLng();
				}

				lat /= (double) (venueIds.size() + 1);
				lng /= (double) (venueIds.size() + 1);
			}

			PointObject p = new PointObject(lat, lng);
			Set<String> venueInArea = new HashSet<>(vo.getNeighbors());
			venueInArea.add(venueId);
			
			AreaObject a = new AreaObject(venueId, p, venueInArea);
			result.put(venueId, a);
		}
		
		return result;
	}
	
	/**
	 * 
	 * @param vMap
	 * @return
	 */
	public static HashMap<String, AreaObject> createSquareCluster(HashMap<String, VenueObject> vMap) {
		HashMap<String, AreaObject> result = new HashMap<>();
		
		RectangleObject coverRectangle = surroundingGrid(vMap.values());
		double scale = 0.1; // the size of each square is 0.1 x 0.1 (latitude and longitude)
		
		PointObject ne = coverRectangle.getNortheast();
		PointObject sw = coverRectangle.getSouthwest();
		
		double base_min_lat = sw.getLat();
		double base_min_lng = sw.getLng();
		
		int numLat = (int) (Math.abs(ne.getLat() - sw.getLat()) / scale);
		int numLng = (int) (Math.abs(ne.getLng() - sw.getLng()) / scale);
		
		// I know it is not a good way to handle this case but maybe it works
		// key is area id; value is set of venue id which is belong to this area
		HashMap<String, Set<String>> venuesInArea = new HashMap<>();
		
		for (String vId : vMap.keySet()) {
			VenueObject vo = vMap.get(vId);
			PointObject loc = vo.getLocation();
			
			// cell id of this venue
			int i = (int) Math.ceil((loc.getLat() - base_min_lat) / scale);
			int j = (int) Math.ceil((loc.getLng() - base_min_lng) / scale);
			
			// area id of venue. Each venue is belong to only 1 area.
			String areaIds = String.valueOf(i * numLat + j);
			
			vo.setAreaId(areaIds);
		}
		
		// create area
		for (int i = 0; i < numLat - 1; i++) {
			for (int j = 0; j < numLng - 1; j++) {
				String areaId = String.valueOf(i * j);
				PointObject sub_ne = new PointObject(base_min_lat + (scale * (double)(i + 1)), base_min_lng + (scale * (double) (j + 1)));
				PointObject sub_sw = new PointObject(base_min_lat + (scale * (double)(i)), base_min_lng + (scale * (double) (j)));
				RectangleObject rObj = new RectangleObject(sub_ne, sub_sw);
				AreaObject area = new AreaObject(areaId, rObj.getCenter(), venuesInArea.get(areaId));
				result.put(areaId, area);
			}
		}
				
		return result;
	}
	
	/**
	 * find the surrounding grid for all of venues
	 * @return the rectangle object which covers all venues.
	 */
	public static RectangleObject surroundingGrid(Collection<VenueObject> vSet){
		double minLng = Double.MAX_VALUE;
		double minLat = Double.MAX_VALUE;
		double maxLat = Double.MIN_VALUE;
		double maxLng = Double.MIN_VALUE;
		
		for (VenueObject c : vSet){
			PointObject point = c.getLocation();
			double lat = point.getLat();
			double lng = point.getLng();
			if (lat > maxLat)
				maxLat = lat;
			if (lng > maxLng)
				maxLng = lng;
			if (lat < minLat)
				minLat = lat;
			if (lng < minLng)
				minLng = lng;
		}
		
		PointObject ne = new PointObject(Utils.roundUp(maxLat, 1), Utils.roundUp(maxLng, 1));
		PointObject sw = new PointObject(Utils.roundDown(minLat, 1), Utils.roundDown(minLng, 1));
		
		return new RectangleObject(ne, sw);
	}

	public static RectangleObject surroundingGrid1(Collection<PointObject> vSet){
		double minLng = Double.MAX_VALUE;
		double minLat = Double.MAX_VALUE;
		double maxLat = Double.MIN_VALUE;
		double maxLng = Double.MIN_VALUE;
		
		for (PointObject point : vSet){
			double lat = point.getLat();
			double lng = point.getLng();
			if (lat > maxLat)
				maxLat = lat;
			if (lng > maxLng)
				maxLng = lng;
			if (lat < minLat)
				minLat = lat;
			if (lng < minLng)
				minLng = lng;
		}
		
		PointObject ne = new PointObject(Utils.roundUp(maxLat, 1), Utils.roundUp(maxLng, 1));
		PointObject sw = new PointObject(Utils.roundDown(minLat, 1), Utils.roundDown(minLng, 1));
		
		return new RectangleObject(ne, sw);
	}
	
	
	
	public static RectangleObject surroundingGrid(ArrayList<String> vSet){
		double minLng = Double.MAX_VALUE;
		double minLat = Double.MAX_VALUE;
		double maxLat = Double.MIN_VALUE;
		double maxLng = Double.MIN_VALUE;
		
		for (String c : vSet){
			PointObject point = new PointObject(c);
			double lat = point.getLat();
			double lng = point.getLng();
			if (lat > maxLat)
				maxLat = lat;
			if (lng > maxLng)
				maxLng = lng;
			if (lat < minLat)
				minLat = lat;
			if (lng < minLng)
				minLng = lng;
		}
		
		PointObject ne = new PointObject(Utils.roundUp(maxLat, 1), Utils.roundUp(maxLng, 1));
		PointObject sw = new PointObject(Utils.roundDown(minLat, 1), Utils.roundDown(minLng, 1));
		
		return new RectangleObject(ne, sw);
	}
		

}
