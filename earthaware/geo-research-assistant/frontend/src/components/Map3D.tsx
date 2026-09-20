import { useEffect, useRef, useState } from "react";
import maplibregl, { type GeoJSONSource, type LngLatLike, type PointLike } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { SelectedAreaPayload } from "../types";

interface Props {
  onAreaSelected: (area: SelectedAreaPayload) => void;
}

const SATELLITE_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    esri: {
      type: "raster",
      tiles: ["https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
      tileSize: 256,
      attribution: "Esri, Maxar, Earthstar Geographics",
      maxzoom: 19,
    },
  },
  layers: [
    {
      id: "esri-imagery",
      type: "raster",
      source: "esri",
      paint: {
        "raster-resampling": "linear",
      },
    },
  ],
};

function centerFromCoords(coords: number[][]) {
  let minLng = Infinity;
  let minLat = Infinity;
  let maxLng = -Infinity;
  let maxLat = -Infinity;
  coords.forEach(([lng, lat]) => {
    minLng = Math.min(minLng, lng);
    minLat = Math.min(minLat, lat);
    maxLng = Math.max(maxLng, lng);
    maxLat = Math.max(maxLat, lat);
  });
  return {
    lat: (minLat + maxLat) / 2,
    lng: (minLng + maxLng) / 2,
    bounds: [
      [minLng, minLat],
      [maxLng, maxLat],
    ] as [[number, number], [number, number]],
  };
}

function rectangleFromCorners(a: { lng: number; lat: number }, b: { lng: number; lat: number }): number[][] {
  const minLng = Math.min(a.lng, b.lng);
  const minLat = Math.min(a.lat, b.lat);
  const maxLng = Math.max(a.lng, b.lng);
  const maxLat = Math.max(a.lat, b.lat);
  return [
    [minLng, minLat],
    [maxLng, minLat],
    [maxLng, maxLat],
    [minLng, maxLat],
    [minLng, minLat],
  ];
}

function makePolygonFeature(coords: number[][]): GeoJSON.FeatureCollection {
  return {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        properties: {},
        geometry: {
          type: "Polygon",
          coordinates: [coords],
        },
      },
    ],
  };
}

function makeEmptyFeatureCollection(): GeoJSON.FeatureCollection {
  return {
    type: "FeatureCollection",
    features: [],
  };
}

function toPoint(p: PointLike): { x: number; y: number } {
  if (Array.isArray(p)) return { x: p[0], y: p[1] };
  const pt = p as { x: number; y: number };
  return { x: pt.x, y: pt.y };
}

function captureSelectionImage(
  map: maplibregl.Map,
  start: { x: number; y: number },
  end: { x: number; y: number }
): { imageDataUrl?: string; error?: string } {
  const canvas = map.getCanvas();
  const dprX = canvas.width / Math.max(1, canvas.clientWidth);
  const dprY = canvas.height / Math.max(1, canvas.clientHeight);

  const minX = Math.max(0, Math.min(start.x, end.x));
  const minY = Math.max(0, Math.min(start.y, end.y));
  const maxX = Math.max(start.x, end.x);
  const maxY = Math.max(start.y, end.y);

  const wCss = Math.max(1, maxX - minX);
  const hCss = Math.max(1, maxY - minY);

  if (wCss < 24 || hCss < 24) return { error: "Selection too small. Draw a larger box." };

  const sx = Math.floor(minX * dprX);
  const sy = Math.floor(minY * dprY);
  const sw = Math.floor(wCss * dprX);
  const sh = Math.floor(hCss * dprY);

  if (sw <= 0 || sh <= 0) return { error: "Invalid capture dimensions." };

  const out = document.createElement("canvas");
  out.width = sw;
  out.height = sh;
  const ctx = out.getContext("2d");
  if (!ctx) return { error: "Canvas context unavailable." };
  try {
    ctx.drawImage(canvas, sx, sy, sw, sh, 0, 0, sw, sh);
    return { imageDataUrl: out.toDataURL("image/jpeg", 0.9) };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { error: `Image capture failed: ${msg}` };
  }
}

export default function Map3D({ onAreaSelected }: Props) {
  const mapRef = useRef<maplibregl.Map | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [mapError, setMapError] = useState<string>("");

  const boxStartRef = useRef<{ lng: number; lat: number } | null>(null);
  const boxStartPxRef = useRef<{ x: number; y: number } | null>(null);
  const isBoxDraggingRef = useRef(false);
  const suppressNextClickRef = useRef(false);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: SATELLITE_STYLE,
      center: [77.1025, 28.7041] as LngLatLike,
      zoom: 5,
      pitch: 50,
      bearing: -15,
      antialias: true,
      maxPitch: 85,
      preserveDrawingBuffer: true,
    });

    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");

    map.on("error", (evt) => {
      const msg = evt.error?.message || "Map render/style error.";
      setMapError(msg);
    });

    map.on("load", () => {
      if (!map.getSource("selection-box")) {
        map.addSource("selection-box", {
          type: "geojson",
          data: makeEmptyFeatureCollection(),
        });
      }

      if (!map.getLayer("selection-box-fill")) {
        map.addLayer({
          id: "selection-box-fill",
          type: "fill",
          source: "selection-box",
          paint: {
            "fill-color": "#00d4ff",
            "fill-opacity": 0.16,
          },
        });
      }

      if (!map.getLayer("selection-box-line")) {
        map.addLayer({
          id: "selection-box-line",
          type: "line",
          source: "selection-box",
          paint: {
            "line-color": "#00d4ff",
            "line-width": 2,
          },
        });
      }

      setMapError("");
    });

    map.on("mousedown", (e) => {
      const mouseEvent = e.originalEvent as MouseEvent;
      if (!mouseEvent.shiftKey) return;

      isBoxDraggingRef.current = true;
      boxStartRef.current = { lng: e.lngLat.lng, lat: e.lngLat.lat };
      boxStartPxRef.current = toPoint(e.point);
      suppressNextClickRef.current = true;
      map.dragPan.disable();
    });

    map.on("mousemove", (e) => {
      if (!isBoxDraggingRef.current || !boxStartRef.current) return;
      const coords = rectangleFromCorners(boxStartRef.current, { lng: e.lngLat.lng, lat: e.lngLat.lat });
      const src = map.getSource("selection-box") as GeoJSONSource | undefined;
      if (src) src.setData(makePolygonFeature(coords));
    });

    map.on("mouseup", (e) => {
      if (!isBoxDraggingRef.current || !boxStartRef.current || !boxStartPxRef.current) return;

      const coords = rectangleFromCorners(boxStartRef.current, { lng: e.lngLat.lng, lat: e.lngLat.lat });
      const { lat, lng, bounds } = centerFromCoords(coords);
      const endPx = toPoint(e.point);
      const capture = captureSelectionImage(map, boxStartPxRef.current, endPx);

      const src = map.getSource("selection-box") as GeoJSONSource | undefined;
      if (src) src.setData(makePolygonFeature(coords));

      onAreaSelected({
        coordinates: coords,
        center: { lat, lng },
        areaType: "polygon",
        imageDataUrl: capture.imageDataUrl,
        imageCaptureError: capture.error,
      });

      map.fitBounds(bounds, { padding: 30, duration: 700 });

      isBoxDraggingRef.current = false;
      boxStartRef.current = null;
      boxStartPxRef.current = null;
      map.dragPan.enable();
    });

    map.on("click", (e) => {
      if (suppressNextClickRef.current) {
        suppressNextClickRef.current = false;
        return;
      }

      const area: SelectedAreaPayload = {
        coordinates: [[e.lngLat.lng, e.lngLat.lat]],
        center: { lat: e.lngLat.lat, lng: e.lngLat.lng },
        areaType: "point",
      };

      const src = map.getSource("selection-box") as GeoJSONSource | undefined;
      if (src) src.setData(makeEmptyFeatureCollection());

      map.flyTo({
        center: [e.lngLat.lng, e.lngLat.lat],
        duration: 800,
        zoom: Math.max(12, map.getZoom()),
      });
      onAreaSelected(area);
    });

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [onAreaSelected]);

  return (
    <div className="relative h-full w-full rounded-xl border border-cyan-400/20">
      <div ref={containerRef} className="h-full w-full rounded-xl" />
      <div className="pointer-events-none absolute bottom-3 left-3 rounded-md border border-cyan-400/30 bg-slate-950/70 px-3 py-1.5 text-xs text-cyan-100">
        Shift + Drag: Draw analysis box | Click: Point analysis
      </div>
      {mapError ? (
        <div className="pointer-events-none absolute left-3 top-3 max-w-[90%] rounded-md border border-red-400/40 bg-red-950/70 px-3 py-2 text-xs text-red-200">
          {mapError}
        </div>
      ) : null}
    </div>
  );
}
