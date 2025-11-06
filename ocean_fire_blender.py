# Fire + Water Ceiling + Music Reactive (Blender 4.5.3 LTS)
# --- FIX 8: Maximum Fire Realism Enhancements ---
# 1. Increased DOMAIN_RESOLUTION to 256.
# 2. Enabled d.use_noise and increased d.vorticity to 1.5.
# 3. Tweaked material to use sharper flame power (3.5), lower minimum density (0.08), and higher Blackbody Intensity (4.0).
# 4. Increased CYCLES_SAMPLES to 256.

import bpy, os, math, wave
import mathutils
from mathutils import Vector
from concurrent.futures import ProcessPoolExecutor

# ====================== USER SETTINGS ======================
AUDIO_PATH = r"/home/jordan/Documents/Blender_scripts/10_sec.wav"
OUTPUT_DIR = r"/home/jordan/Desktop/"
OUTPUT_FILE = "fire_ocean_visible.mp4"

RES_X, RES_Y = 256,256
FPS = 23

SCENE_DURATION_PAD = None
SCENE_MIN_SECONDS = 6.0
SIM_WARMUP_FRAMES = 96

# --- REALISM BOOST: INCREASED FIDELITY AND SAMPLES ---
DOMAIN_RESOLUTION = 256
CYCLES_SAMPLES   = 256

# Cycles volume stepping tuned so emission shows
VOLUME_STEP_RATE = 0.02
VOLUME_MAX_STEPS = 1024
VOLUME_BOUNCES   = 3

USE_TRUE_EMISSIVE_LIGHTING = True

# Mixer / Sequencer behavior
USE_VSE_FOR_RENDER = False

# Music responsiveness
EMISSION_BASE  = 1.0
EMISSION_SCALE = 12.0

# Camera ellipse
CAM_A = 3.2
CAM_E = 0.30
CAM_HEIGHT = 0.6
CAM_KEY_STEP = 3

# Scene layout
FLOOR_Z   = -0.05
CEILING_Z = 4.0

# Water shading & motion
BUMP_STRENGTH_BASE  = 0.40
MAPPING_DRIFT_X     = 0.015
MAPPING_DRIFT_Y     = 0.012

# Sparks (upward)
SPARK_RATE       = 900
SPARK_LIFETIME_S = 2.0
SPARK_SIZE       = 0.010
SPARK_UP_SPEED   = 2.0
SPARKS_MAX_TOTAL = 200_000
WIND_STRENGTH    = 80
WIND_NOISE       = 0.5
SPARK_EMISSION_STRENGTH = 120.0

# Dynamic Paint (ripples)
WAVE_SPEED   = 1.0
WAVE_DAMPING = 0.015
WAVE_HEIGHT  = 0.45
WAVE_RADIUS  = 0.10

# Density behavior:
DENSITY_MODE = "flame_masked"

# --- FIRE MATERIAL REALISM TWEAKS ---
FIRE_FLAME_POWER = 3.5
SMOKE_DENSITY_POWER = 2.0
BLACKBODY_INTENSITY = 4.0
MINIMUM_SMOKE_DENSITY = 0.08

# --- SIMULATION REALISM TWEAKS ---
DOMAIN_VORTICITY = 1.5

# --- LIGHTING FIXES FOR SPARKS/RIPPLES ---
CEILING_LIGHT_POWER = 1500.0

# --- Follow light rig (broad, soft, cinematic) ---
FOLLOW_LIGHT_ENABLE   = True
FOLLOW_LIGHT_DIST     = 3.0
FOLLOW_SOFT_SIZE      = 8.0
FOLLOW_SOFT_POWER     = 600.0
FOLLOW_RIM_SIZE       = 5.0
FOLLOW_RIM_POWER      = 300.0
FOLLOW_COLOR          = (1.0, 0.98, 0.95)
WORLD_MIN_LIFT        = 0.05

# --- Baking (DISABLED: Relying on REPLAY mode to prevent crashes) ---
BAKE_CACHE_DIR  = "/home/jordan/Desktop/fire_bake"

# --- Visibility helper ---
FAST_VISIBILITY_MODE = True

# ====================== BASIC UTILS ======================
def _safe_set(obj, attr, value):
    if not hasattr(obj, attr): return
    try:
        setattr(obj, attr, value)
    except TypeError:
        for caster in (int, float):
            try:
                setattr(obj, attr, caster(value)); return
            except Exception:
                pass

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def set_scene_basics():
    s = bpy.context.scene
    s.render.resolution_x = RES_X
    s.render.resolution_y = RES_Y
    s.render.resolution_percentage = 100
    s.render.fps = FPS
    s.view_settings.view_transform = 'Filmic'
    s.view_settings.look = 'High Contrast'
    s.view_settings.exposure = 0.15
    s.world = bpy.data.worlds.new("World")
    s.world.use_nodes = True
    bg = s.world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.01, 0.01, 0.02, 1.0)
    bg.inputs["Strength"].default_value = WORLD_MIN_LIFT

def set_render_output(dirpath, filename):
    os.makedirs(bpy.path.abspath(dirpath), exist_ok=True)
    s = bpy.context.scene
    s.render.filepath = os.path.join(dirpath, filename)
    s.render.image_settings.file_format = 'FFMPEG'
    s.render.ffmpeg.format = 'MPEG4'
    s.render.ffmpeg.codec = 'H264'
    s.render.ffmpeg.audio_codec = 'AAC'
    s.render.ffmpeg.audio_bitrate = 192

def try_enable_cycles():
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    c = s.cycles
    # --- REALISM FIX: INCREASED SAMPLES ---
    c.samples = CYCLES_SAMPLES
    c.volume_step_rate = VOLUME_STEP_RATE
    c.volume_max_steps = VOLUME_MAX_STEPS
    c.volume_bounces = VOLUME_BOUNCES
    c.use_light_tree = True
    c.use_persistent_data = False
    try:
        bpy.context.view_layer.cycles.use_denoising = True
    except:
        pass
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        s.cycles.device = 'GPU' if prefs.compute_device_type != 'NONE' else 'CPU'
    except:
        s.cycles.device = 'CPU'
    if hasattr(c, "use_emission_light") and USE_TRUE_EMISSIVE_LIGHTING:
        c.use_emission_light = True

# ====================== AUDIO ENVELOPE ======================
import struct as _struct

def _bytes_to_samples(block, sw):
    if sw == 1:
        vals = _struct.unpack(f"{len(block)}B", block); return [(b-128)/127.0 for b in vals]
    if sw == 2:
        n = len(block)//2; vals = _struct.unpack(f"<{n}h", block); return [v/32768.0 for v in vals]
    if sw == 3:
        out=[]
        for i in range(0,len(block),3):
            b0,b1,b2=block[i],block[i+1],block[i+2]
            v = b0 | (b1<<8) | (b2<<16)
            if v & 0x800000: v -= 1<<24
            out.append(v/8388608.0)
        return out
    if sw == 4:
        n=len(block)//4; vals=_struct.unpack(f"<{n}i", block); return [v/2147483648.0 for v in vals]
    raise ValueError("Unsupported sample width")

def _rms(xs):
    if not xs: return 0.0
    return math.sqrt(sum(x*x for x in xs)/len(xs))

def _downmix_to_mono(xs,ch):
    if ch==1: return xs
    mono=[]; i=0
    for v in xs:
        if i%ch==0: mono.append(0.0)
        mono[-1]+=v/ch; i+=1
    return mono

def _read_wav_duration_seconds(path):
    with wave.open(path,'rb') as w:
        return w.getnframes()/w.getframerate()

def _rms_frames_worker(args):
    raw,start_f,count_f,frame_bytes,sw,ch=args
    out=[]; base=start_f*frame_bytes
    for i in range(count_f):
        start=base+i*frame_bytes
        block=raw[start:start+frame_bytes]
        if len(block)<frame_bytes:
            block += b"\x00"*(frame_bytes-len(block))
        samples=_bytes_to_samples(block,sw)
        mono=_downmix_to_mono(samples,ch)
        out.append(_rms(mono))
    return start_f,out

def _read_wav_envelope_parallel(path,fps,attack=0.005,release=0.20):
    with wave.open(path,'rb') as w:
        ch=w.getnchannels(); sw=w.getsampwidth(); sr=w.getframerate()
        n=w.getnframes(); raw=w.readframes(n)
    spf=max(1,int(round(sr/float(fps))))
    frame_bytes=spf*sw*ch
    total_frames=(len(raw)+frame_bytes-1)//frame_bytes
    tasks=[]; f=0
    while f<total_frames:
        count=min(900,total_frames-f)
        tasks.append((raw,f,count,frame_bytes,sw,ch)); f+=count
    results=[]
    max_workers=max(1,(os.cpu_count() or 4)-1)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for start,vals in ex.map(_rms_frames_worker,tasks):
            results.append((start,vals))
    results.sort()
    per=[]; [per.extend(v) for _,v in results]
    per=per[:total_frames]
    att=math.exp(-spf/(sr*max(attack,1e-4)))
    rel=math.exp(-spf/(sr*max(release,1e-4)))
    env=0.0; sm=[]
    for rms in per:
        env = att*env + (1-att)*rms if rms>env else rel*env + (1-rel)*rms
        sm.append(env)
    peak=max(sm) if sm else 1.0
    if peak>0: sm=[min(1.0,v/peak) for v in sm]
    return sm

# ====================== CAMERA ======================
def _kepler_solve_E(M,e,iters=6):
    E=M
    for _ in range(iters): E=M+e*math.sin(E)
    return E

def make_orbit_focus_and_aim(flow_obj, ceiling_z, bias=0.35):
    flow_z = flow_obj.matrix_world.translation.z
    aim_z  = flow_z + bias*(ceiling_z - flow_z)
    aim = bpy.data.objects.new("FocusAim", None)
    aim.empty_display_type='PLAIN_AXES'
    aim.location=(0.0,0.0,aim_z)
    bpy.context.scene.collection.objects.link(aim)
    return flow_obj, aim

def animate_camera_ellipse(
    cam,
    focus_obj,
    period_frames,
    a=None,             # semi-major axis; if None we’ll derive it
    e=None,             # eccentricity
    height=None,        # camera height over focus
    step=3,
    start_at_periapsis=True,
    track_target=None,
    use_sun_radius=True,
    sun_radius=None,    # if provided, use this as the ellipse 'a'
    margin=0.35         # extra distance to stay outside domain volume
):
    scn = bpy.context.scene
    f0, f1 = scn.frame_start, scn.frame_end

    # Derive a reasonable domain height and radius for auto choices
    if bpy.data.objects.get("Domain"):
        domain = bpy.data.objects["Domain"]
        bbox = [domain.matrix_world @ mathutils.Vector(corner) for corner in domain.bound_box]
        z_min = min(v.z for v in bbox)
        z_max = max(v.z for v in bbox)
        dom_height = z_max - z_min
        # in XY, approximate a circular bound from cube scaling:
        dom_radius_xy = max(
            (domain.matrix_world.to_scale().x + domain.matrix_world.to_scale().y) * 0.5,
            1.0
        )
    else:
        dom_height = 4.0
        dom_radius_xy = 3.5

    # Eccentricity default (mild ellipse)
    e = 0.30 if e is None else float(e)

    # Height default tracks domain size
    height = (dom_height * 0.45) if height is None else float(height)

    # Semi-major axis 'a'
    # - If sun_radius is provided (or use_sun_radius=True and we find an object named 'SunRadius'),
    #   we’ll honor that; otherwise, we keep you just outside the domain.
    if a is None:
        a = dom_radius_xy + margin

    if use_sun_radius:
        # User-specified sun_radius wins
        if sun_radius is not None:
            a = float(sun_radius)
        else:
            # Optional: look for a helper Empty named "SunRadius" and use its X scale as meters
            sr = bpy.data.objects.get("SunRadius")
            if sr:
                a = max(0.1, sr.scale.x)  # simple convention: scale.x = radius
    # Ensure 'a' stays outside domain a bit
    a = max(a, dom_radius_xy + margin)

    # Compute b, c for ellipse with the fire as a focus
    b = a * math.sqrt(max(1.0 - e * e, 1e-6))
    c = e * a

    # Focus point (the fire / flow) and ellipse center
    focus = focus_obj.matrix_world.translation
    center = focus - Vector((0.0, c, 0.0))  # keep focus at +c on Y

    # Camera lens & target
    cam.data.lens_unit = 'FOV'
    cam.data.angle = math.radians(60.0)
    cam.data.clip_start = 0.01

    target = track_target or focus_obj
    # Remove old TRACK_TO if re-running
    for con in list(cam.constraints):
        if con.type == 'TRACK_TO':
            cam.constraints.remove(con)

    tcon = cam.constraints.new(type='TRACK_TO')
    tcon.target = target
    tcon.track_axis = 'TRACK_NEGATIVE_Z'
    tcon.up_axis = 'UP_Y'

    # Animate ellipse parametric solution with Kepler focus
    N = max(1, int(period_frames))
    M0 = 0.0 if start_at_periapsis else math.pi

    # Clear prior location fcurves so we don’t layer multiple orbits
    if cam.animation_data and cam.animation_data.action:
        for fc in list(cam.animation_data.action.fcurves):
            if fc.data_path == "location":
                cam.animation_data.action.fcurves.remove(fc)

    for f in range(f0, f1 + 1, step):
        t = (f - f0) / float(N)
        M = 2.0 * math.pi * t + M0
        E = _kepler_solve_E(M, e)
        X = a * (math.cos(E) - e)  # along major axis (focus at origin)
        Y = b * math.sin(E)

        # Swap to world XY with focus offset and add height on Z
        cam.location = center + Vector((Y, X, focus.z + height))
        cam.keyframe_insert('location', frame=f)

    # Linear interpolation for smooth speed
    if cam.animation_data and cam.animation_data.action:
        for fc in cam.animation_data.action.fcurves:
            if fc.data_path == "location":
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'


# ====================== FIRE DOMAIN + FLOW ======================
def build_domain_and_flow():
    fire_z = FLOOR_Z + 0.6
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.35, location=(0, 0, fire_z))
    flow = bpy.context.active_object
    flow.name = "Flow"

    # --- Hiding the Flow object starts here ---
    mat_flow = bpy.data.materials.get("Flow_Invisible")
    if not mat_flow:
        mat_flow = bpy.data.materials.new("Flow_Invisible")
        mat_flow.use_nodes = True
        bsdf = mat_flow.node_tree.nodes.get("Principled BSDF") or mat_flow.node_tree.nodes.new('ShaderNodeBsdfPrincipled')

        # --- FORCE TRANSPARENCY AND ZERO EMISSION ---
        bsdf.inputs["Base Color"].default_value = (0,0,0,1)
        emission_input_name = "Emission Color" if "Emission Color" in bsdf.inputs else "Emission"
        if emission_input_name in bsdf.inputs:
             bsdf.inputs[emission_input_name].default_value = (0,0,0,1)
        bsdf.inputs["Emission Strength"].default_value = 0.0
        # Ensure it is completely transparent/transmission
        if "Transmission" in bsdf.inputs: bsdf.inputs["Transmission"].default_value = 1.0
        if "Alpha" in bsdf.inputs: bsdf.inputs["Alpha"].default_value = 0.0

    flow.data.materials.clear()
    flow.data.materials.append(mat_flow)

    # --- GUARANTEED HIDING ---
    flow.hide_render = True
    flow.hide_viewport = False
    # Make the emitter unobtrusive in viewport
    flow.display_type = 'BOUNDS'  # or 'WIRE'
    flow.hide_select = True       # avoid accidentally moving it
    if hasattr(flow, 'cycles_visibility'):
        flow.cycles_visibility.camera = False
    # -------------------------------------------------------------------------

    mf = flow.modifiers.new("Fluid", "FLUID")
    mf.fluid_type = "FLOW"
    fs = mf.flow_settings
    fs.flow_type = "BOTH"
    fs.flow_behavior = "INFLOW"
    fs.flow_source = "MESH"
    fs.surface_distance = 0.0
    fs.temperature = 2.3
    fs.fuel_amount = 2.6
    fs.use_initial_velocity = True
    fs.velocity_normal = 2.0
    if hasattr(fs, "velocity"):
        try: fs.velocity = (0.0, 0.0, 1.8)
        except Exception:
            try:
                fs.velocity[0] = 0.0; fs.velocity[1] = 0.0; fs.velocity[2] = 1.8
            except Exception: pass
    elif hasattr(fs, "velocity_factor"):
        try: fs.velocity_factor = 1.8
        except Exception: pass
    else:
        for ax, val in (("velocity_x", 0.0), ("velocity_y", 0.0), ("velocity_z", 1.8)):
            if hasattr(fs, ax): _safe_set(fs, ax, val)

    # Domain sized to contain flames
    flame_top = fire_z + 3.5
    z_min = FLOOR_Z - 0.5
    z_max = max(flame_top, CEILING_Z + 0.3)
    domain_center_z = (z_min + z_max) / 2
    domain_height = z_max - z_min
    domain_radius = CAM_A + 1.2

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, domain_center_z))
    domain = bpy.context.active_object
    domain.name = "Domain"
    domain.scale = (domain_radius, domain_radius, domain_height / 2)
    bpy.ops.object.transform_apply(scale=True, location=False, rotation=False)

    md = domain.modifiers.new("Fluid", "FLUID")
    md.fluid_type = "DOMAIN"
    d = md.domain_settings
    d.domain_type = "GAS"
    # --- REALISM FIX: INCREASED RESOLUTION ---
    d.resolution_max = DOMAIN_RESOLUTION
    d.use_adaptive_domain = True
    # --- REALISM FIX: ADDED NOISE AND VORTICITY ---
    d.use_noise = True
    d.vorticity = DOMAIN_VORTICITY

    # --- FORCING REPLAY MODE TO AVOID BAKE CRASH ---
    d.cache_type = 'REPLAY'
    d.time_scale = 0.85
    d.vorticity = 1.2
    d.use_noise = False
    # Set cache directory for consistency, even in REPLAY
    cache_dir = bpy.path.abspath(BAKE_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    if hasattr(d, "cache_directory"): d.cache_directory = cache_dir

    # OVERRIDING THE CONFLICTING LINES ABOVE WITH THE REALISM FIXES
    d.vorticity = DOMAIN_VORTICITY
    d.use_noise = True

    print(f"Domain adjusted: height={domain_height:.2f}, radius={domain_radius:.2f}, centerZ={domain_center_z:.2f}")
    return domain, flow

# ====================== FIRE MATERIAL (music + sinusoidal warmth) ======================
def make_fire_material(domain, ctrl):
    def _in(node, name, idx=0):
        try: return node.inputs[name]
        except KeyError: return node.inputs[idx]
    def _out(node, name, idx=0):
        try: return node.outputs[name]
        except KeyError: return node.outputs[idx]

    mat = bpy.data.materials.new("FireDomain")
    mat.use_nodes = True
    domain.data.materials.clear()
    domain.data.materials.append(mat)

    nt = mat.node_tree; n = nt.nodes; l = nt.links
    for x in list(n): n.remove(x)

    out = n.new("ShaderNodeOutputMaterial"); out.location = (1100, 0)
    vol = n.new("ShaderNodeVolumePrincipled"); vol.location = (800, 0)
    l.new(_out(vol,"Volume"), _in(out,"Volume"))

    vinf = n.new("ShaderNodeVolumeInfo"); vinf.location = (-720, 0)

    # flame^FIRE_FLAME_POWER + 0.02
    powf = n.new("ShaderNodeMath"); powf.operation = 'POWER'; powf.location = (-520, 60)
    # --- REALISM FIX: SHARPER FLAME FALLOFF ---
    powf.inputs[1].default_value = FIRE_FLAME_POWER
    l.new(_out(vinf, "Flame"), _in(powf, 0, 0))

    add_floor = n.new("ShaderNodeMath"); add_floor.operation = 'ADD'; add_floor.location = (-340, 60)
    add_floor.inputs[1].default_value = 0.02
    l.new(_out(powf, "Value"), _in(add_floor, 0, 0))

    # Music gain: (1 + m * EMISSION_SCALE)
    mg = n.new("ShaderNodeMath"); mg.operation='MULTIPLY'; mg.location=(420,180)
    mg.inputs[0].default_value = EMISSION_BASE
    drv = mg.inputs[1].driver_add('default_value').driver; drv.type='SCRIPTED'
    v = drv.variables.new(); v.name = 'm'; v.targets[0].id = ctrl; v.targets[0].data_path = '["music_amp"]'
    drv.expression = f"(1.0 + m*{EMISSION_SCALE})"

    gain = n.new("ShaderNodeMath"); gain.operation='MULTIPLY'; gain.location=(620,120)
    gain.inputs[0].default_value = 200.0
    l.new(_out(mg, "Value"), _in(gain, 1, 1))

    fstr = n.new("ShaderNodeMath"); fstr.operation='MULTIPLY'; fstr.location=(820,120)
    l.new(_out(add_floor, "Value"), _in(fstr, 0, 0))
    l.new(_out(gain, "Value"), _in(fstr, 1, 1))

    # Temperature base (0..1) -> Kelvin
    clampT = n.new("ShaderNodeClamp"); clampT.location = (-520, -160)
    clampT.inputs["Min"].default_value = 0.0
    clampT.inputs["Max"].default_value = 1.2
    l.new(_out(vinf, "Temperature"), _in(clampT, "Value", 0))

    scaleT = n.new("ShaderNodeMath"); scaleT.operation='MULTIPLY'; scaleT.location=(-340,-160)
    scaleT.inputs[1].default_value = 900.0
    l.new(_out(clampT, "Result"), _in(scaleT, 0, 0))

    addK = n.new("ShaderNodeMath"); addK.operation='ADD'; addK.location=(-160,-160)
    addK.inputs[1].default_value = 1400.0
    l.new(_out(scaleT, "Value"), _in(addK, 0, 0))

    # Smooth sinusoidal warmth shift: 120 * sin(2*pi*(frame/FPS)*0.33)
    val120 = n.new("ShaderNodeValue"); val120.location = (40, -220); val120.outputs[0].default_value = 120.0
    sinK = n.new("ShaderNodeMath"); sinK.operation='SINE'; sinK.location = (220, -220)
    ang = sinK.inputs[0].driver_add('default_value').driver
    ang.type='SCRIPTED'
    vF = ang.variables.new(); vF.name='f'; vF.targets[0].id_type='SCENE'; vF.targets[0].id=bpy.context.scene; vF.targets[0].data_path='frame_current'
    ang.expression = f"2*3.14159265*(f/{FPS}*0.33)"
    mulS = n.new("ShaderNodeMath"); mulS.operation='MULTIPLY'; mulS.location = (400, -220)
    l.new(_out(sinK,"Value"), mulS.inputs[0])
    l.new(val120.outputs[0], mulS.inputs[1])
    addWarm = n.new("ShaderNodeMath"); addWarm.operation='ADD'; addWarm.location = (580, -180)
    l.new(_out(addK,"Value"), addWarm.inputs[0])
    l.new(mulS.outputs[0], addWarm.inputs[1])

    # Density
    if DENSITY_MODE == "flame_masked":
        powd = n.new("ShaderNodeMath"); powd.operation='POWER'; powd.location=(-520,-20)
        # --- REALISM FIX: THINNER SMOKE MASK ---
        powd.inputs[1].default_value = SMOKE_DENSITY_POWER
        l.new(_out(vinf, "Flame"), _in(powd, 0, 0))
        clampd = n.new("ShaderNodeClamp"); clampd.location=(-340,-20)
        l.new(_out(powd, "Value"), _in(clampd, "Value", 0))
        muld = n.new("ShaderNodeMath"); muld.operation='MULTIPLY'; muld.location=(720,-40)
        muld.inputs[0].default_value = 0.02
        l.new(_out(clampd, "Result"), _in(muld, 1, 1))
        add_min = n.new("ShaderNodeMath"); add_min.operation='ADD'; add_min.location=(880,-40)
        # --- REALISM FIX: LOWER MINIMUM DENSITY ---
        add_min.inputs[1].default_value = MINIMUM_SMOKE_DENSITY
        l.new(_out(muld, "Value"), _in(add_min, 0, 0))
        dens_socket = add_min.outputs["Value"]
    else:
        const_d = n.new("ShaderNodeValue"); const_d.location = (720, -40)
        const_d.outputs[0].default_value = 0.03
        dens_socket = const_d.outputs[0]

    # Principled Volume hookups
    if "Blackbody Intensity" in vol.inputs:
        # --- REALISM FIX: HIGHER BLACKBODY INTENSITY ---
        vol.inputs["Blackbody Intensity"].default_value = BLACKBODY_INTENSITY
    if "Temperature" in vol.inputs:
        l.new(addWarm.outputs[0], vol.inputs["Temperature"])
    if "Emission Strength" in vol.inputs:
        l.new(_out(fstr, "Value"), vol.inputs["Emission Strength"])
    if "Emission Color" in vol.inputs:
        vol.inputs["Emission Color"].default_value = (1.0, 0.55, 0.18, 1.0)
    if "Density" in vol.inputs:
        l.new(dens_socket, vol.inputs["Density"])

    # --- ENSURED MINIMUM DENSITY IS SET ---
    if FAST_VISIBILITY_MODE:
        try:
            vol.inputs["Emission Strength"].default_value = max(220.0, vol.inputs["Emission Strength"].default_value)
            # This line ensures a minimum density of 0.08 is set
            vol.inputs["Density"].default_value = MINIMUM_SMOKE_DENSITY
        except Exception:
            pass

# ====================== SPARKS RENDER OBJECT ======================
def make_spark_object():
    # Create an Icosphere and move it far away so it doesn't appear in the scene
    bpy.ops.mesh.primitive_ico_sphere_add(radius=SPARK_SIZE*1.5, location=(1000, 1000, 1000))
    spark_obj = bpy.context.active_object
    spark_obj.name = "SparkObject"
    # CRITICAL: This hides the object itself from render, but allows its instances (the sparks) to render.
    spark_obj.hide_render = True

    # Simple Emissive Material
    mat = bpy.data.materials.new(name="SparkMat")
    mat.use_nodes = True
    nt = mat.node_tree; n = nt.nodes; l = nt.links
    bsdf = nt.nodes.get("Principled BSDF")

    # Make it a bright, self-illuminating red-orange sphere
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1.0, 0.2, 0.0, 1.0)

        emission_input_name = "Emission Color" if "Emission Color" in bsdf.inputs else "Emission"

        if emission_input_name in bsdf.inputs:
            bsdf.inputs[emission_input_name].default_value = (1.0, 0.2, 0.0, 1.0)

        bsdf.inputs["Emission Strength"].default_value = SPARK_EMISSION_STRENGTH

    spark_obj.data.materials.append(mat)

    return spark_obj
# ==============================================================================

# ====================== CEILING LIGHT (NEW) ======================
def add_ceiling_spotlight(z_pos):
    light_data = bpy.data.lights.new("CeilingSpot", type='AREA')
    light_data.shape = 'SQUARE'
    light_data.size = 12.0 # Large area to cover the scene
    light_data.energy = CEILING_LIGHT_POWER
    light_data.color = (0.7, 0.8, 1.0) # Slight cool color to contrast the warm fire

    light = bpy.data.objects.new("CeilingSpot", light_data)
    bpy.context.scene.collection.objects.link(light)

    # Position slightly above the ceiling, pointing straight down
    light.location = (0.0, 0.0, z_pos + 0.5)
    light.rotation_euler = (math.radians(180), 0.0, 0.0) # Rotate 180 degrees on X to point down

    print(f"Added Ceiling Spotlight at Z={z_pos + 0.5:.2f} with {CEILING_LIGHT_POWER}W energy.")
    return light

# ====================== WATER ======================
def _setup_principled_water(bsdf):
    names=[i.name for i in bsdf.inputs]
    if "Base Color" in names: bsdf.inputs["Base Color"].default_value=(0.02,0.07,0.12,1.0)
    if "Transmission" in names:
        bsdf.inputs["Transmission"].default_value=1.0
        if "IOR" in names: bsdf.inputs["IOR"].default_value=1.333
        if "Roughness" in names: bsdf.inputs["Roughness"].default_value=0.06
        return
    if "Transmission Weight" in names:
        bsdf.inputs["Transmission Weight"].default_value=1.0
    if "Specular IOR Level" in names:
        bsdf.inputs["Specular IOR Level"].default_value=1.0
    if "IOR" in names: bsdf.inputs["IOR"].default_value=1.333
    if "Roughness" in names: bsdf.inputs["Roughness"].default_value=0.06

def make_water_material(name="Water"):
    m=bpy.data.materials.new(name); m.use_nodes=True
    nt=m.node_tree; n=nt.nodes; l=nt.links
    for x in list(n): n.remove(x)
    out=n.new("ShaderNodeOutputMaterial"); out.location=(900,0)
    bsdf=n.new("ShaderNodeBsdfPrincipled"); bsdf.location=(650,0); l.new(bsdf.outputs[0], out.inputs["Surface"])
    _setup_principled_water(bsdf)
    texcoord=n.new("ShaderNodeTexCoord"); mapping=n.new("ShaderNodeMapping")
    mapping.location=(-700,-200); texcoord.location=(-900,-200)
    try: l.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
    except: pass
    noise_big=n.new("ShaderNodeTexNoise"); noise_big.location=(-500,0);   noise_big.inputs["Scale"].default_value=2.2
    noise_fine=n.new("ShaderNodeTexNoise"); noise_fine.location=(-500,-220); noise_fine.inputs["Scale"].default_value=9.5
    mix=n.new("ShaderNodeMixRGB"); mix.blend_type='ADD'; mix.location=(-250,-120)
    l.new(noise_big.outputs["Fac"], mix.inputs[1]); l.new(noise_fine.outputs["Fac"], mix.inputs[2])
    bump=n.new("ShaderNodeBump"); bump.location=(350,-120)
    bump.inputs["Strength"].default_value=BUMP_STRENGTH_BASE
    l.new(mix.outputs["Color"], bump.inputs["Height"])
    if "Normal" in bsdf.inputs: l.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    for axis,speed in enumerate((MAPPING_DRIFT_X,MAPPING_DRIFT_Y,0.0)):
        try:
            d=mapping.inputs["Location"].driver_add("default_value",axis).driver
            d.type='SCRIPTED'; v=d.variables.new(); v.name='f'
            v.targets[0].id_type='SCENE'; v.targets[0].id=bpy.context.scene; v.targets[0].data_path='frame_current'
            d.expression=f"{speed}*f"
        except: pass
    return m

# ====================== SPARKS + RIPPLE IMPACTS ======================
def build_sparks_and_ripples(ceiling_obj, spark_obj): # Added spark_obj parameter
    import contextlib
    scn=bpy.context.scene
    @contextlib.contextmanager
    def _active(obj):
        prev=bpy.context.view_layer.objects.active
        try:
            bpy.context.view_layer.objects.active=obj
            for o in bpy.context.selected_objects: o.select_set(False)
            obj.select_set(True); yield
        finally:
            if prev and prev.name in bpy.context.view_layer.objects:
                bpy.context.view_layer.objects.active=prev

    # emitter
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.1, location=(0,0,0.6))
    emitter=bpy.context.active_object; emitter.name="SparkEmitter"

    # particle system
    ps_mod=emitter.modifiers.new("Sparks",'PARTICLE_SYSTEM')
    psys=emitter.particle_systems[-1]; p=psys.settings
    duration_s=(scn.frame_end-scn.frame_start+1)/scn.render.fps
    p.type='EMITTER'
    p.count=int(min(SPARK_RATE*duration_s, SPARKS_MAX_TOTAL))
    p.frame_start=scn.frame_start; p.frame_end=scn.frame_end
    p.lifetime=max(1,int(SPARK_LIFETIME_S*scn.render.fps))
    p.emit_from='VOLUME'; p.use_modifier_stack=True
    if hasattr(p,"particle_size"): p.particle_size=SPARK_SIZE
    p.normal_factor=SPARK_UP_SPEED
    if hasattr(p,"object_align_factor"): p.object_align_factor=(0.0,0.0,1.5)
    p.effector_weights.gravity=0.2

    # --- RENDER SPARKS AS THE INSTANCE OBJECT ---
    p.render_type = 'OBJECT'
    p.instance_object = spark_obj
    # --------------------------------------------

    # IMPORTANT: no collision on ceiling (avoids dependency cycle)
    if hasattr(p,"use_die_on_collision"): p.use_die_on_collision=False

    # wind
    bpy.ops.object.effector_add(type='WIND', location=(0,0,1.0))
    wind=bpy.context.active_object; wind.field.strength=WIND_STRENGTH; wind.field.noise=WIND_NOISE

    # ceiling DP canvas (no Collision modifier here!)
    subs=ceiling_obj.modifiers.new("Subsurf","SUBSURF")
    subs.levels=3; subs.render_levels=3

    with _active(ceiling_obj):
        bpy.ops.object.modifier_add(type='DYNAMIC_PAINT')
        dpm=ceiling_obj.modifiers[-1]; dpm.name="DP_Canvas"; dpm.ui_type='CANVAS'
        canvas=dpm.canvas_settings
        if canvas is None:
            try: bpy.ops.dpaint.type_toggle(type='CANVAS'); canvas=dpm.canvas_settings
            except Exception: pass
        if canvas and hasattr(canvas,"canvas_surfaces"):
            bpy.ops.dpaint.surface_slot_add()
            surf=canvas.canvas_surfaces[-1]
            if hasattr(surf,"surface_type"): surf.surface_type='WAVE'
            for k,v in dict(wave_speed=WAVE_SPEED,wave_damping=WAVE_DAMPING,wave_height=WAVE_HEIGHT).items():
                if hasattr(surf,k): setattr(surf,k,v)
            if hasattr(surf,"use_wave_open_border"): surf.use_wave_open_border=True

    with _active(emitter):
        bpy.ops.object.modifier_add(type='DYNAMIC_PAINT')
        dpb=emitter.modifiers[-1]; dpb.name="DP_Brush"; dpb.ui_type='BRUSH'
        brush=dpb.brush_settings
        if brush is None:
            try: bpy.ops.dpaint.type_toggle(type='BRUSH'); brush=dpb.brush_settings
            except Exception: brush=None
        if brush:
            if hasattr(brush,"paint_source"): brush.paint_source='PARTICLE_SYSTEM'
            elif hasattr(brush,"paint_source_type"): brush.paint_source_type='PARTICLE_SYSTEM'
            if hasattr(brush,"particle_system"): brush.particle_system=psys
            if hasattr(brush,"use_absolute_alpha"): brush.use_absolute_alpha=True
            if hasattr(brush,"wave_factor"): brush.wave_factor=1.0
            if hasattr(brush,"radius"): brush.radius=WAVE_RADIUS

    return emitter, psys, wind

# ====================== VSE HELPER ======================
def ensure_vse_has_scene_strip():
    scn = bpy.context.scene
    scn.render.use_sequencer = bool(USE_VSE_FOR_RENDER)
    if not USE_VSE_FOR_RENDER:
        return
    seq = scn.sequence_editor if scn.sequence_editor else scn.sequence_editor_create()
    has_scene = any(s.type == 'SCENE' for s in seq.sequences_all)
    if not has_scene:
        strip = seq.sequences.new_scene(
            name="SceneRender",
            scene=scn,
            channel=1,
            frame_start=scn.frame_start
        )
        strip.frame_final_duration = scn.frame_end - scn.frame_start + 1
    for s in seq.sequences_all:
        if s.type == 'SOUND' and s.channel <= 1:
            s.channel = 2

# ====================== FOLLOW LIGHT RIG ======================
def add_follow_light_rig(cam):
    if not FOLLOW_LIGHT_ENABLE or cam is None:
        return []
    col = bpy.context.scene.collection
    made = []

    key_data = bpy.data.lights.new("FollowKey", type='AREA')
    key_data.shape = 'SQUARE'
    key_data.size  = FOLLOW_SOFT_SIZE
    key_data.energy = FOLLOW_SOFT_POWER
    key_data.color  = FOLLOW_COLOR
    if hasattr(key_data, "specular_factor"):
        key_data.specular_factor = 0.35
    key = bpy.data.objects.new("FollowKey", key_data)
    col.objects.link(key)
    key.parent = cam
    key.matrix_parent_inverse = cam.matrix_world.inverted()
    key.location = (0.0, 0.0, FOLLOW_LIGHT_DIST)
    key.rotation_euler = (0.0, 0.0, 0.0)
    if hasattr(key_data, "shadow_soft_size"):
        key_data.shadow_soft_size = FOLLOW_SOFT_SIZE * 0.08
    made.append(key)

    rim_data = bpy.data.lights.new("FollowRim", type='AREA')
    rim_data.shape = 'SQUARE'
    rim_data.size  = FOLLOW_RIM_SIZE
    rim_data.energy = FOLLOW_RIM_POWER
    rim_data.color  = FOLLOW_COLOR
    if hasattr(rim_data, "specular_factor"):
        rim_data.specular_factor = 0.25
    rim = bpy.data.objects.new("FollowRim", rim_data)
    col.objects.link(rim)
    rim.parent = cam
    rim.matrix_parent_inverse = cam.matrix_world.inverted()
    rim.location = (0.0, FOLLOW_RIM_SIZE*0.35, FOLLOW_LIGHT_DIST*0.85)
    rim.rotation_euler = (math.radians(-15.0), 0.0, 0.0)
    if hasattr(rim_data, "shadow_soft_size"):
        rim_data.shadow_soft_size = FOLLOW_RIM_SIZE * 0.08
    made.append(rim)

    for o in made:
        if hasattr(o, "cycles_visibility"):
            o.cycles_visibility.camera = False
    return made

# ====================== MUSIC CONTROL ======================
def add_music_and_control(audio_path):
    scn=bpy.context.scene
    scn.sequence_editor_create(); seq=scn.sequence_editor
    snd_len_frames = int(round(_read_wav_duration_seconds(audio_path) * scn.render.fps))
    if snd_len_frames <= 0: snd_len_frames = scn.frame_end - scn.frame_start + 1
    frame_cursor=scn.frame_start; loop_idx=0
    while frame_cursor <= scn.frame_end:
        seq.sequences.new_sound(f"Music_{loop_idx}", audio_path, 2, frame_cursor)
        frame_cursor += snd_len_frames; loop_idx += 1

    ctrl=bpy.data.objects.new("MusicControl", None)
    ctrl.empty_display_type='SPHERE'; ctrl.location=(0,0,2.5)
    scn.collection.objects.link(ctrl)

    env_one=_read_wav_envelope_parallel(audio_path, scn.render.fps)
    need=scn.frame_end - scn.frame_start + 1
    env=[env_one[i%len(env_one)] for i in range(need)] if env_one else [0.0]*need

    f0,f1=scn.frame_start,scn.frame_end
    frames=list(range(f0,f1+1,4))
    vals=[float(env[i-f0]) for i in frames]

    a=ctrl.animation_data_create(); a.action=bpy.data.actions.new("MusicEnv")
    fc=a.action.fcurves.new('["music_amp"]'); fc.keyframe_points.add(len(frames))
    for i,(f,v) in enumerate(zip(frames,vals)):
        kp=fc.keyframe_points[i]; kp.co=(f,v); kp.interpolation='LINEAR'
    ctrl["music_amp"]=vals[0] if vals else 0.0
    return ctrl

# ====================== MAIN ======================
SAVE_BLEND_PATH = "/home/jordan/Desktop/fire_water_music.blend"
MAKE_PATHS_RELATIVE = True
EXIT_AFTER_SAVE = False

def main():
    # Base scene + engine
    clear_scene()
    set_scene_basics()
    try_enable_cycles()

    # Timing
    assert os.path.isfile(AUDIO_PATH), f"Missing audio {AUDIO_PATH}"
    dur=_read_wav_duration_seconds(AUDIO_PATH)
    pad=SCENE_DURATION_PAD if SCENE_DURATION_PAD is not None else max(0.5, min(3.0, dur*0.1))
    total_seconds=max(dur+pad, SCENE_MIN_SECONDS)

    scn=bpy.context.scene
    scn.frame_start=1
    scn.frame_end=int(total_seconds*FPS)
    print(f"Blender {bpy.app.version_string}")
    print(f"Audio: {dur:.2f}s | Total: {total_seconds:.2f}s | Frames: {scn.frame_end - scn.frame_start + 1}")

    # Fire (domain + flow) FIRST
    domain,flow=build_domain_and_flow()

    # Camera (can be present during bake)
    cam_data=bpy.data.cameras.new("Camera"); cam=bpy.data.objects.new("Camera", cam_data)
    scn.collection.objects.link(cam); scn.camera=cam
    orbit_focus, aim_target = make_orbit_focus_and_aim(flow, CEILING_Z, bias=0.35)
    animate_camera_ellipse(
        cam,
        orbit_focus,
        bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1,
        e=CAM_E,
        height=CAM_HEIGHT,
        step=CAM_KEY_STEP,
        track_target=aim_target,
        use_sun_radius=True,
        sun_radius= 6.0
    )

    # Output path configured
    set_render_output(OUTPUT_DIR, OUTPUT_FILE)

    # --- MUSIC CONTROL + FIRE MATERIAL ---
    ctrl=add_music_and_control(AUDIO_PATH)
    make_fire_material(domain, ctrl)

    # Sequencer stays off for render
    scn.render.use_sequencer = False
    ensure_vse_has_scene_strip()

    # ====================== STATIC AND DYNAMIC OBJECTS ======================
    # Water floor & ceiling
    bpy.ops.mesh.primitive_plane_add(size=24, location=(0,0,FLOOR_Z))
    floor=bpy.context.active_object; floor.name="WaterFloor"
    floor.data.materials.append(make_water_material("WaterFloorMat"))

    bpy.ops.mesh.primitive_plane_add(size=24, location=(0,0,CEILING_Z))
    ceiling=bpy.context.active_object; ceiling.name="WaterCeiling"
    ceiling.data.materials.append(make_water_material("WaterCeilingMat"))

    # --- CREATE THE OBJECT THAT WILL BE RENDERED AS SPARKS ---
    spark_obj = make_spark_object()

    # Sparks + ripples (now pass the spark object)
    build_sparks_and_ripples(ceiling, spark_obj)

    # Follow light rig (soft)
    add_follow_light_rig(cam)

    # --- ADD POWERFUL CEILING SPOTLIGHT ---
    add_ceiling_spotlight(CEILING_Z)


    # ====================== FLUID EVALUATION (REPLAY Mode) ======================
    # Warm up sim so voxels exist (for REPLAY evaluation)
    scn.frame_current = min(scn.frame_start + max(12, SIM_WARMUP_FRAMES + 18), scn.frame_end)
    for f in range(scn.frame_start, scn.frame_current+1):
        # This frame loop forces the simulation to calculate data frame-by-frame (REPLAY mode)
        scn.frame_set(f)

    # Keep cam below ceiling
    if cam.location.z >= CEILING_Z - 0.5:
        cam.location.z = CEILING_Z - 0.5

    # Camera hold during warmup
    hold_until = scn.frame_start + SIM_WARMUP_FRAMES
    if cam.animation_data and cam.animation_data.action:
        for fc in cam.animation_data.action.fcurves:
            if fc.keyframe_points:
                first_val=fc.keyframe_points[0].co[1]
                kp=fc.keyframe_points.insert(frame=hold_until, value=first_val)
                kp.interpolation='CONSTANT'
                for k in fc.keyframe_points[1:]: k.interpolation='LINEAR'

    # ---- Save .blend (optional) ----
    if SAVE_BLEND_PATH:
        path = bpy.path.abspath(SAVE_BLEND_PATH)
        print(f"Saving scene to: {path}")
        if MAKE_PATHS_RELATIVE:
            try: bpy.ops.file.make_paths_relative()
            except Exception: pass
        bpy.ops.wm.save_as_mainfile(filepath=path)
        print(f"Scene saved to {path}")
        if EXIT_AFTER_SAVE:
            return

    print("Starting render:", bpy.path.abspath(os.path.join(OUTPUT_DIR, OUTPUT_FILE)))
    bpy.ops.render.render(animation=True, write_still=False)
    print("Render complete.")

if __name__=="__main__":
    main()
