import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// In-browser GLB viewer (orbit). The GLB comes from the scene compiler via Blender's glTF exporter.
export function GlbViewer({ url }: { url: string }) {
  const host = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState("loading…");

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      setStatus("WebGL is not available in this browser.");
      return;
    }
    const width = el.clientWidth || 640;
    const height = 420;
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf2f2f2);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x888888, 2.0));
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(5, 8, 5);
    scene.add(sun);
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.05, 500);
    const controls = new OrbitControls(camera, renderer.domElement);
    let frame = 0;
    new GLTFLoader().load(
      url,
      (gltf) => {
        scene.add(gltf.scene);
        const box = new THREE.Box3().setFromObject(gltf.scene);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3()).length();
        camera.position.copy(center).add(new THREE.Vector3(size * 0.6, size * 0.7, size * 0.6));
        controls.target.copy(center);
        controls.update();
        setStatus("");
      },
      undefined,
      () => setStatus("Could not load the 3D model."),
    );
    const tick = () => {
      frame = requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, camera);
    };
    tick();
    return () => {
      cancelAnimationFrame(frame);
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, [url]);

  return (
    <div className="card">
      <div ref={host} className="viewer" data-testid="glb-viewer" />
      {status && <div className="muted">{status}</div>}
    </div>
  );
}
