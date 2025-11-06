# Fire + Water Ceiling + Music Reactive (Blender 4.5.3 LTS) — fixed VSE render + options
# Visible flames with NO extra lights:
# - Domain REPLAY (no bake), Flow=BOTH (flame+temperature)
# - Principled Volume: Emission from flame^2.8 * big_gain * music
# - Blackbody from temperature (Kelvin), Density=0.03 (always visible)  [or flame-masked]
# - Tighter Cycles volume marching so emission pops
# - Audio loops to fill 6s shot; short warmup & camera hold
# - Water floor/ceiling + sparks hitting ceiling to make ripples
#
# CHANGES:
# - ensure_vse_has_scene_strip(): prevents black renders when audio strips exist
# - USE_VSE_FOR_RENDER toggle
# - DENSITY_MODE: "constant" (original) or "flame_masked" (clearer air)

import bpy, os, math, wave, struct
import mathutils
from mathutils import Vector
from concurrent.futures import ProcessPoolExecutor

import re, time
import threading

bpy.context.scene.render.use_sequencer = False

# ====================== USER SETTINGS ======================
AUDIO_PATH = r"/home/jordan/Downloads/entangled4.wav"
OUTPUT_DIR = r"/home/jordan/Desktop/"
OUTPUT_FILE = "fire_ocean_visible.mp4"

RES_X, RES_Y = 256, 256
FPS = 23

SCENE_DURATION_PAD = None
SCENE_MIN_SECONDS = 6.0
SIM_WARMUP_FRAMES = 36

DOMAIN_RESOLUTION = 48
CYCLES_SAMPLES   = 128

# Cycles volume stepping tuned so emission shows
VOLUME_STEP_RATE = 0.75   # tighter than 1.0
VOLUME_MAX_STEPS = 512
VOLUME_BOUNCES   = 3

USE_TRUE_EMISSIVE_LIGHTING = True   # let emission cast light
USE_FAKE_FIRE_LIGHT        = False  # keep OFF – we want to prove no lights needed

# Mixer / Sequencer behavior
USE_VSE_FOR_RENDER = False   # keep True if you want audio embedded in the mp4

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

# Dynamic Paint (ripples)
WAVE_SPEED   = 1.0
WAVE_DAMPING = 0.015
WAVE_HEIGHT  = 0.45
WAVE_RADIUS  = 0.10

# Volume density behavior for visibility when camera is inside the domain:
#   "constant"    -> original: density = 0.03 everywhere (always visible)
#   "flame_masked"-> density = 0.03 * clamp(flame^1.2, 0..1) (clearer air)
DENSITY_MODE = "constant"   # "constant" | "flame_masked"

# --- Baking (turn this on to precompute the sim to disk) ---
USE_BAKED_SIM   = False
BAKE_CACHE_DIR  = "/home/jordan/Desktop/fire_bake"  # any writable folder
BAKE_USE_NOISE  = False      # True = bake extra detail (slower, looks nicer)
BAKE_FREE_OLD   = True       # delete old cache before baking
# Show a live progress bar while baking
USE_BAKE_PROGRESS = True

# ====================== UTILS ======================
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

def _count_vdb_frames(cache_dir: str) -> int:
    """Recursively count .vdb frames written by the gas sim."""
    total = 0
    for root, _, files in os.walk(cache_dir):
        for fn in files:
            if fn.lower().endswith(".vdb"):
                total += 1
    return total

class WM_OT_bake_with_progress(bpy.types.Operator):
    """Bake fluid with a polling progress bar based on VDB files."""
    bl_idname = "wm.bake_with_progress"
    bl_label = "Bake With Progress (Gas)"
    bl_options = {'REGISTER'}

    domain_name: bpy.props.StringProperty()
    cache_dir: bpy.props.StringProperty()
    frame_start: bpy.props.IntProperty()
    frame_end: bpy.props.IntProperty()
    use_noise: bpy.props.BoolProperty(default=False)

    _timer = None
    _started = False
    _last_count = 0

    def modal(self, context, event):
        wm = context.window_manager
        if event.type == 'TIMER':
            # poll cache
            done = _count_vdb_frames(self.cache_dir)
            total_frames = max(1, self.frame_end - self.frame_start + 1)
            # heuristic: at least one .vdb per frame → clamp
            pct = min(1.0, done / float(total_frames))
            wm.progress_update(int(pct * 100))
            context.workspace.status_text_set(
                f"Baking Gas: {int(pct*100)}%  ({done}/{total_frames} files)"
            )
            if self._last_count != done:
                print(f"[Bake] progress: {done}/{total_frames} ({pct*100:.1f}%)")
                self._last_count = done

            # finish when bake finished & files stopped growing
            if done >= total_frames and self._started is True:
                self._finish(context)
                self.report({'INFO'}, "Bake complete.")
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def _finish(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        wm.progress_end()
        context.workspace.status_text_set(None)

    def execute(self, context):
        wm = context.window_manager
        wm.progress_begin(0, 100)
        self._timer = wm.event_timer_add(0.5, window=context.window)

        # start the real bake on the next tick so UI stays responsive
        def _kick_bake():
            try:
                domain = bpy.data.objects.get(self.domain_name)
                if not domain:
                    print("[Bake] ERROR: domain not found")
                    return None
                _set_active(domain)
                # DATA bake
                print("[Bake] Baking DATA…")
                bpy.ops.fluid.bake_data()
                # NOISE bake if requested
                if self.use_noise:
                    print("[Bake] Baking NOISE…")
                    bpy.ops.fluid.bake_noise()
            except Exception as e:
                print("[Bake] ERROR:", e)
            finally:
                self._started = True
            return None  # do not repeat
        bpy.app.timers.register(_kick_bake, first_interval=0.1)

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

def register_bake_progress_op():
    if not hasattr(bpy.types, "WM_OT_bake_with_progress"):
        bpy.utils.register_class(WM_OT_bake_with_progress)

def _set_active(obj):
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)

def _toggle_all_drivers(enable: bool):
    """Mute/unmute all drivers across common datablocks to avoid depsgraph edits during bake."""
    datablocks = []
    datablocks.extend(bpy.data.objects)
    datablocks.extend(bpy.data.materials)
    datablocks.extend(bpy.data.node_groups)
    datablocks.extend(bpy.data.worlds)
    for datablock in datablocks:
        ad = getattr(datablock, "animation_data", None)
        if not ad or not getattr(ad, "drivers", None):
            continue
        for fcu in ad.drivers:
            try:
                fcu.mute = (not enable)
            except Exception:
                pass

def bake_domain(domain):
    """
    MODULAR bake with all drivers muted during the bake.
    Avoids 'Dependency graph update requested during evaluation'.
    """
    scn = bpy.context.scene

    bpy.context.view_layer.objects.active = domain
    for o in bpy.context.selected_objects:
        o.select_set(False)
    domain.select_set(True)

    md = domain.modifiers.get("Fluid")
    assert md and md.fluid_type == 'DOMAIN', "Domain must have a Fluid DOMAIN modifier"
    d = md.domain_settings

    cache_dir = bpy.path.abspath(BAKE_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)

    if hasattr(d, "cache_type"):          d.cache_type = 'MODULAR'
    if hasattr(d, "cache_directory"):     d.cache_directory = cache_dir
    if hasattr(d, "cache_frame_start"):   d.cache_frame_start = scn.frame_start
    if hasattr(d, "cache_frame_end"):     d.cache_frame_end   = scn.frame_end
    if hasattr(d, "use_noise"):           d.use_noise = bool(BAKE_USE_NOISE)
    if hasattr(d, "time_scale"):          d.time_scale = 0.85
    if hasattr(d, "use_adaptive_domain"): d.use_adaptive_domain = True

    if BAKE_FREE_OLD:
        try:
            bpy.ops.fluid.free_all()
        except Exception as e:
            print("free_all() warning:", e)

    scn.frame_set(scn.frame_start)
    print(f"[Bake] (MODULAR) Cache: {cache_dir}")
    _toggle_all_drivers(False)
    try:
        try:
            with bpy.context.temp_override(scene=scn,
                                           view_layer=bpy.context.view_layer,
                                           active_object=domain,
                                           selected_objects=[domain]):
                bpy.ops.fluid.bake_data()
                if BAKE_USE_NOISE:
                    bpy.ops.fluid.bake_noise()
        except AttributeError:
            override = {'scene': scn,
                        'view_layer': bpy.context.view_layer,
                        'active_object': domain,
                        'selected_objects': [domain]}
            bpy.ops.fluid.bake_data(override)
            if BAKE_USE_NOISE:
                bpy.ops.fluid.bake_noise(override)
    finally:
        _toggle_all_drivers(True)
        bpy.context.view_layer.update()
        print(f"[Bake] Done. Cache at: {cache_dir}")

# ====================== AUDIO HELPERS ======================
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

# ====================== SCENE BASICS ======================
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def set_scene_basics():
    s=bpy.context.scene
    s.render.resolution_x=RES_X; s.render.resolution_y=RES_Y; s.render.resolution_percentage=100
    s.render.fps=FPS
    s.view_settings.view_transform='Filmic'
    s.view_settings.look='High Contrast'
    s.view_settings.exposure = 0.15
    s.world=bpy.data.worlds.new("World"); s.world.use_nodes=True
    bg=s.world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value=(0.01,0.01,0.02,1.0)
    bg.inputs["Strength"].default_value=0.02

def set_render_output(dirpath,filename):
    os.makedirs(bpy.path.abspath(dirpath), exist_ok=True)
    s=bpy.context.scene
    s.render.filepath=os.path.join(dirpath, filename)
    s.render.image_settings.file_format='FFMPEG'
    s.render.ffmpeg.format='MPEG4'
    s.render.ffmpeg.codec='H264'
    s.render.ffmpeg.audio_codec='AAC'
    s.render.ffmpeg.audio_bitrate=192

def try_enable_cycles():
    s=bpy.context.scene
    s.render.engine='CYCLES'
    c=s.cycles
    c.samples=CYCLES_SAMPLES
    c.volume_step_rate=VOLUME_STEP_RATE
    c.volume_max_steps=VOLUME_MAX_STEPS
    c.volume_bounces=VOLUME_BOUNCES
    c.use_light_tree=True
    c.use_persistent_data=False
    try: bpy.context.view_layer.cycles.use_denoising=True
    except: pass
    try:
        prefs=bpy.context.preferences.addons['cycles'].preferences
        s.cycles.device='GPU' if prefs.compute_device_type!='NONE' else 'CPU'
    except: s.cycles.device='CPU'
    if hasattr(c,"use_emission_light") and USE_TRUE_EMISSIVE_LIGHTING:
        c.use_emission_light=True

# ====================== CAMERA: Ellipse + Aim ======================
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

def animate_camera_ellipse(cam, focus_obj, period_frames,
                           a=None, e=None, height=None, step=3,
                           start_at_periapsis=True, track_target=None):
    """
    Improved version: automatically adjusts camera distance and height
    based on domain or flow height to keep the fire fully in view.
    """

    scn = bpy.context.scene
    f0, f1 = scn.frame_start, scn.frame_end

    if bpy.data.objects.get("Domain"):
        domain = bpy.data.objects["Domain"]
        bbox = [domain.matrix_world @ mathutils.Vector(corner) for corner in domain.bound_box]
        z_min = min(v.z for v in bbox)
        z_max = max(v.z for v in bbox)
        dom_height = z_max - z_min
    else:
        dom_height = 4.0

    a = a or (CAM_A + dom_height * 0.5)
    e = e or 0.25
    height = height or (dom_height * 0.45)

    b = a * math.sqrt(1.0 - e * e)
    c = e * a
    focus = focus_obj.matrix_world.translation
    center = focus - Vector((0.0, c, 0.0))

    cam.data.lens_unit = 'FOV'
    cam.data.angle = math.radians(60.0)
    cam.data.clip_start = 0.01

    target = track_target or focus_obj
    tcon = cam.constraints.new(type='TRACK_TO')
    tcon.target = target
    tcon.track_axis = 'TRACK_NEGATIVE_Z'
    tcon.up_axis = 'UP_Y'

    N = max(1, int(period_frames))
    M0 = 0.0 if start_at_periapsis else math.pi
    for f in range(f0, f1 + 1, step):
        t = (f - f0) / float(N)
        M = 2.0 * math.pi * t + M0
        E = _kepler_solve_E(M, e)
        X = a * (math.cos(E) - e)
        Y = b * math.sin(E)
        cam.location = center + Vector((Y, X, focus.z + height))
        cam.keyframe_insert('location', frame=f)

    if cam.animation_data and cam.animation_data.action:
        for fc in cam.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'

    print(f"Camera orbit adjusted: distance={a:.2f}, height={height:.2f}, FOV=60°")

# ====================== BUILD: FIRE (REPLAY, BOTH) ======================
def build_domain_and_flow():
    fire_z = FLOOR_Z + 0.6
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.35, location=(0, 0, fire_z))
    flow = bpy.context.active_object
    flow.name = "Flow"

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
        try:
            fs.velocity = (0.0, 0.0, 1.8)
        except Exception:
            try:
                fs.velocity[0] = 0.0
                fs.velocity[1] = 0.0
                fs.velocity[2] = 1.8
            except Exception:
                pass
    elif hasattr(fs, "velocity_factor"):
        try:
            fs.velocity_factor = 1.8
        except Exception:
            try:
                fs.velocity_factor[0] = 0.0
                fs.velocity_factor[1] = 0.0
                fs.velocity_factor[2] = 1.8
            except Exception:
                pass
    else:
        for ax, val in (("velocity_x", 0.0), ("velocity_y", 0.0), ("velocity_z", 1.8)):
            if hasattr(fs, ax):
                _safe_set(fs, ax, val)

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
    d.resolution_max = DOMAIN_RESOLUTION
    d.use_adaptive_domain = True
    d.cache_type = "REPLAY"
    d.time_scale = 0.85
    d.vorticity = 1.2
    d.use_noise = False

    print(f"Domain adjusted: height={domain_height:.2f}, radius={domain_radius:.2f}, centerZ={domain_center_z:.2f}")
    return domain, flow

# ====================== FIRE MATERIAL ======================
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

    a_flame = n.new("ShaderNodeAttribute"); a_flame.attribute_name = "flame"; a_flame.location = (-700,60)
    a_temp  = n.new("ShaderNodeAttribute"); a_temp.attribute_name  = "temperature"; a_temp.location = (-700,-160)

    powf = n.new("ShaderNodeMath"); powf.operation = 'POWER'; powf.location = (-480,60)
    powf.inputs[1].default_value = 2.8
    l.new(_out(a_flame,"Fac"), _in(powf,0,0))

    add_floor = n.new("ShaderNodeMath"); add_floor.operation='ADD'; add_floor.location=(-300,60)
    add_floor.inputs[1].default_value = 0.02
    l.new(_out(powf,"Value"), _in(add_floor,0,0))

    mg = n.new("ShaderNodeMath"); mg.operation='MULTIPLY'; mg.location=(420,180)
    mg.inputs[0].default_value=EMISSION_BASE
    drv=mg.inputs[1].driver_add('default_value').driver; drv.type='SCRIPTED'
    v=drv.variables.new(); v.name='m'; v.targets[0].id=ctrl; v.targets[0].data_path='["music_amp"]'
    drv.expression=f"(1.0 + m*{EMISSION_SCALE})"

    gain = n.new("ShaderNodeMath"); gain.operation='MULTIPLY'; gain.location=(620,120)
    gain.inputs[0].default_value=80.0
    l.new(_out(mg,"Value"), _in(gain,1,1))

    fstr = n.new("ShaderNodeMath"); fstr.operation='MULTIPLY'; fstr.location=(820,120)
    l.new(_out(add_floor,"Value"), _in(fstr,0,0))
    l.new(_out(gain,"Value"), _in(fstr,1,1))

    clamp = n.new("ShaderNodeClamp"); clamp.location=(-480,-160)
    clamp.inputs["Min"].default_value=0.0
    clamp.inputs["Max"].default_value=1.2
    l.new(_out(a_temp,"Fac"), _in(clamp,"Value",0))

    scale = n.new("ShaderNodeMath"); scale.operation='MULTIPLY'; scale.location=(-300,-160)
    scale.inputs[1].default_value=900.0
    l.new(_out(clamp,"Result"), _in(scale,0,0))

    addk = n.new("ShaderNodeMath"); addk.operation='ADD'; addk.location=(-120,-160)
    addk.inputs[1].default_value=1400.0
    l.new(_out(scale,"Value"), _in(addk,0,0))

    powd = n.new("ShaderNodeMath"); powd.operation='POWER'; powd.location=(-480,-20)
    powd.inputs[1].default_value=1.2
    l.new(_out(a_flame,"Fac"), _in(powd,0,0))
    clampd = n.new("ShaderNodeClamp"); clampd.location=(-300,-20)
    l.new(_out(powd,"Value"), _in(clampd,"Value",0))
    muld = n.new("ShaderNodeMath"); muld.operation='MULTIPLY'; muld.location=(720,-40)
    muld.inputs[0].default_value=0.02
    l.new(_out(clampd,"Result"), _in(muld,1,1))
    add_min = n.new("ShaderNodeMath"); add_min.operation='ADD'; add_min.location=(880,-40)
    add_min.inputs[1].default_value=0.006
    l.new(_out(muld,"Value"), _in(add_min,0,0))

    if "Blackbody Intensity" in vol.inputs: vol.inputs["Blackbody Intensity"].default_value = 3.0
    if "Temperature" in vol.inputs:         l.new(_out(addk,"Value"), vol.inputs["Temperature"])
    if "Emission Strength" in vol.inputs:   l.new(_out(fstr,"Value"), vol.inputs["Emission Strength"])
    if "Emission Color" in vol.inputs:      vol.inputs["Emission Color"].default_value=(1.0,0.55,0.18,1.0)
    if "Density" in vol.inputs:             l.new(_out(add_min,"Value"), vol.inputs["Density"])

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

# ====================== WATER (v1/v2 safe) ======================
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
def build_sparks_and_ripples(ceiling_obj):
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

    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.1, location=(0,0,0.6))
    emitter=bpy.context.active_object; emitter.name="SparkEmitter"
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
    if hasattr(p,"use_die_on_collision"): p.use_die_on_collision=True

    bpy.ops.object.effector_add(type='WIND', location=(0,0,1.0))
    wind=bpy.context.active_object; wind.field.strength=WIND_STRENGTH; wind.field.noise=WIND_NOISE

    subs=ceiling_obj.modifiers.new("Subsurf","SUBSURF")
    subs.levels=3; subs.render_levels=3
    ceiling_obj.modifiers.new("Collision",'COLLISION')

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

# ====================== VSE: ensure scene strip so renders aren't black ======================
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

# ====================== MAIN ======================
SAVE_BLEND_PATH = "/home/jordan/Desktop/fire_water_music.blend"
MAKE_PATHS_RELATIVE = True
EXIT_AFTER_SAVE = False  # set True if you only want to generate the .blend, not render

def main():
    clear_scene()
    set_scene_basics()
    try_enable_cycles()

    assert os.path.isfile(AUDIO_PATH), f"Missing audio {AUDIO_PATH}"
    dur=_read_wav_duration_seconds(AUDIO_PATH)
    pad=SCENE_DURATION_PAD if SCENE_DURATION_PAD is not None else max(0.5, min(3.0, dur*0.1))
    total_seconds=max(dur+pad, SCENE_MIN_SECONDS)

    scn=bpy.context.scene
    scn.frame_start=1
    scn.frame_end=int(total_seconds*FPS)
    print(f"Blender {bpy.app.version_string}")
    print(f"Audio: {dur:.2f}s | Total: {total_seconds:.2f}s | Frames: {scn.frame_end - scn.frame_start + 1}")

    # Fire
    domain,flow=build_domain_and_flow()

    # Camera
    cam_data=bpy.data.cameras.new("Camera"); cam=bpy.data.objects.new("Camera", cam_data)
    scn.collection.objects.link(cam); scn.camera=cam
    orbit_focus, aim_target = make_orbit_focus_and_aim(flow, CEILING_Z, bias=0.35)
    animate_camera_ellipse(cam, orbit_focus, scn.frame_end-scn.frame_start+1,
                           a=CAM_A, e=CAM_E, height=CAM_HEIGHT, step=CAM_KEY_STEP,
                           track_target=aim_target)

    # Push camera slightly outside domain to avoid being inside the volume
    if bpy.data.objects.get("Domain"):
        domain = bpy.data.objects["Domain"]
        bbox = [domain.matrix_world @ mathutils.Vector(corner) for corner in domain.bound_box]
        x_min, x_max = min(v.x for v in bbox), max(v.x for v in bbox)
        y_min, y_max = min(v.y for v in bbox), max(v.y for v in bbox)
        z_min, z_max = min(v.z for v in bbox), max(v.z for v in bbox)
        cam.location.y = y_min - 0.5
        print(f"Camera moved behind domain: new Y = {cam.location.y:.2f}")

    # Output
    set_render_output(OUTPUT_DIR, OUTPUT_FILE)

    # Music + material
    ctrl=add_music_and_control(AUDIO_PATH)
    make_fire_material(domain, ctrl)

    # Water floor & ceiling
    bpy.ops.mesh.primitive_plane_add(size=24, location=(0,0,FLOOR_Z))
    floor=bpy.context.active_object; floor.name="WaterFloor"
    floor.data.materials.append(make_water_material("WaterFloorMat"))

    bpy.ops.mesh.primitive_plane_add(size=24, location=(0,0,CEILING_Z))
    ceiling=bpy.context.active_object; ceiling.name="WaterCeiling"
    ceiling.data.materials.append(make_water_material("WaterCeilingMat"))

    # Sparks + ripples
    build_sparks_and_ripples(ceiling)

    # ----------------- F12 sanity patch (drop-in) -----------------
    scn = bpy.context.scene
    scn.render.use_sequencer = False
    scn.render.engine = 'CYCLES'

    hot = scn.frame_start + max(12, SIM_WARMUP_FRAMES + 18)
    scn.frame_current = min(hot, scn.frame_end)
    scn.frame_set(scn.frame_current)

    for obj in bpy.data.objects:
        obj.hide_render = False
    for col in bpy.data.collections:
        col.hide_render = False

    dom = bpy.data.objects.get("Domain")
    cam = scn.camera
    if dom and cam:
        from mathutils import Vector
        bb = [dom.matrix_world @ Vector(c) for c in dom.bound_box]
        x_min,x_max = min(v.x for v in bb), max(v.x for v in bb)
        y_min,y_max = min(v.y for v in bb), max(v.y for v in bb)
        z_min,z_max = min(v.z for v in bb), max(v.z for v in bb)
        cam.location = ((x_min+x_max)*0.5, y_min - 1.0, (z_min+z_max)*0.55)
        cam.data.lens_unit = 'FOV'
        cam.data.angle = math.radians(60)
        cam.data.clip_start = 0.01
        cam.data.clip_end = 2000.0
        tgt = bpy.data.objects.get("FocusAim") or dom
        if not any(c.type=='TRACK_TO' for c in cam.constraints):
            c = cam.constraints.new(type='TRACK_TO')
            c.target = tgt
            c.track_axis = 'TRACK_NEGATIVE_Z'
            c.up_axis = 'UP_Y'

        # --- BLACK FRAME KILLER (force-visible sanity) ---
        # 1) Freeze camera & put it farther outside, staring at domain center
        if cam.animation_data:
            cam.animation_data_clear()
        dom_cx = 0.5*(x_min + x_max); dom_cy = 0.5*(y_min + y_max); dom_cz = 0.5*(z_min + z_max)
        cam.location = (dom_cx, y_min - 3.0, dom_cz)
        cam.data.lens_unit = 'FOV'
        cam.data.angle = math.radians(60)
        cam.data.clip_start = 0.01
        cam.data.clip_end = 5000.0
        if not any(c.type == 'TRACK_TO' for c in cam.constraints):
            cns = cam.constraints.new(type='TRACK_TO')
            cns.target = tgt
            cns.track_axis = 'TRACK_NEGATIVE_Z'
            cns.up_axis = 'UP_Y'

        # 2) Overpowering temporary volume so it cannot be black
        mat = dom.data.materials[0] if dom.data.materials else bpy.data.materials.new("TmpVol")
        if not dom.data.materials:
            dom.data.materials.append(mat)
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes): nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
        vol = nt.nodes.new("ShaderNodeVolumePrincipled"); vol.location = (350, 0)
        nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
        vol.inputs["Emission Color"].default_value = (1.0, 0.8, 0.3, 1.0)
        vol.inputs["Emission Strength"].default_value = 2000.0
        vol.inputs["Density"].default_value = 0.08
        if "Blackbody Intensity" in vol.inputs:
            vol.inputs["Blackbody Intensity"].default_value = 0.0

        # 3) Render safety toggles (tonemapping / world / volume steps)
        scn.view_settings.view_transform = 'Filmic'
        scn.view_settings.look = 'None'
        scn.view_settings.exposure = 0.5
        try:
            scn.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.2
        except Exception:
            pass
        cy = scn.cycles
        cy.samples = max(64, cy.samples)
        cy.volume_step_rate = min(0.5, cy.volume_step_rate)
        cy.volume_max_steps = max(2048, getattr(cy, "volume_max_steps", 512))
        cy.volume_bounces = max(3, getattr(cy, "volume_bounces", 1))
        try: cy.use_emission_light = True
        except: pass

        # 4) Tiny emissive test cube – proves the path if visible
        bpy.ops.mesh.primitive_cube_add(size=0.2, location=(dom_cx, dom_cy, z_min + 0.2))
        cube = bpy.context.active_object
        emit_mat = bpy.data.materials.new("TEST_Emit")
        emit_mat.use_nodes = True
        nt2 = emit_mat.node_tree
        for n in list(nt2.nodes): nt2.nodes.remove(n)
        o2 = nt2.nodes.new("ShaderNodeOutputMaterial"); e2 = nt2.nodes.new("ShaderNodeEmission")
        nt2.links.new(e2.outputs[0], o2.inputs[0])
        e2.inputs["Strength"].default_value = 50.0
        cube.data.materials.append(emit_mat)
        # --- end BLACK FRAME KILLER ---

    print("[F12 sanity] frame", scn.frame_current, "ready")
    # ------------------------------------------------------------

    # Pre-roll REPLAY sim so voxels exist even in background
    seed_until = scn.frame_current
    for f in range(scn.frame_start, seed_until + 1):
        scn.frame_set(f)

    # Keep cam below ceiling
    if cam.location.z >= CEILING_Z - 0.5: cam.location.z = CEILING_Z - 0.5

    # Camera hold during warmup so ignition isn't a pan
    hold_until = scn.frame_start + SIM_WARMUP_FRAMES
    if cam.animation_data and cam.animation_data.action:
        for fc in cam.animation_data.action.fcurves:
            if fc.keyframe_points:
                first_val=fc.keyframe_points[0].co[1]
                kp=fc.keyframe_points.insert(frame=hold_until, value=first_val)
                kp.interpolation='CONSTANT'
                for k in fc.keyframe_points[1:]: k.interpolation='LINEAR'

    ensure_vse_has_scene_strip()

    # ---- SAVE .BLEND BEFORE RENDER ----
    if SAVE_BLEND_PATH:
        path = bpy.path.abspath(SAVE_BLEND_PATH)
        print(f"Saving scene to: {path}")
        if MAKE_PATHS_RELATIVE:
            try:
                bpy.ops.file.make_paths_relative()
            except Exception:
                print("Could not make paths relative.")
        bpy.ops.wm.save_as_mainfile(filepath=path)
        print(f"Scene saved to {path}")
        if EXIT_AFTER_SAVE:
            print("Exiting after save (EXIT_AFTER_SAVE=True)")
            return

    scn = bpy.context.scene
    scn.render.use_sequencer = False
    scn.frame_current = min(scn.frame_start + max(12, SIM_WARMUP_FRAMES + 18), scn.frame_end)
    scn.frame_set(scn.frame_current)

    print("Starting render:", bpy.path.abspath(os.path.join(OUTPUT_DIR, OUTPUT_FILE)))
    bpy.ops.render.render(animation=True, write_still=False)
    print("Render complete.")

if __name__=="__main__":
    main()
