#!/usr/bin/env python3
"""Check a generated wheel model and optionally render its review artifacts.

    python -X utf8 scripts/inspect_galileo_x.py generated/galileo_x_<timestamp> --render

Always re-run after changing dimensions or sitting angles. Failure exits nonzero
and writes a diagnostic report; this tool never changes the robot's parameters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from draft.generation.wheel_leg_inspection import inspect_sitting


def render_artifacts(folder: Path, output: Path, result, params: dict) -> None:
    # Rendering is optional: the CPU geometry check does not need plotting/GL.
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model = mujoco.MjModel.from_xml_path(str(folder / 'scene.xml'))
    data = mujoco.MjData(model)
    width, height = 1200, 800
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.headlight.ambient[:] = [0.45] * 3
    model.vis.headlight.diffuse[:] = [0.8] * 3
    model.vis.headlight.specular[:] = [0.15] * 3
    font_path = Path('C:/Windows/Fonts/segoeui.ttf')
    font = ImageFont.truetype(str(font_path), 23) if font_path.exists() else ImageFont.load_default()

    def caption(pixels, text):
        picture = Image.fromarray(pixels)
        draw = ImageDraw.Draw(picture)
        draw.rectangle([0, 0, width, 48], fill=(247, 249, 252))
        draw.text((22, 9), text, font=font, fill=(35, 49, 70))
        return picture

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-0.12, 0, 0.265]
    camera.azimuth, camera.elevation, camera.distance = 90, 0, 1.4
    standing = model.key_qpos[model.key('standing').id].copy()
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        def frame(qpos, title, perspective=False):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            model.vis.global_.orthographic = 0 if perspective else 1
            model.vis.global_.fovy = 45 if perspective else 0.86
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render().copy()
            if not perspective:
                # A horizontal orthographic camera sees the plane edge-on.
                # Render a mask so the background and the z=0 line remain clear.
                renderer.enable_segmentation_rendering()
                background = renderer.render()[:, :, 0] < 0
                renderer.disable_segmentation_rendering()
                pixels[background] = [247, 249, 252]
                floor_y = round(height * (0.5 + camera.lookat[2] / model.vis.global_.fovy))
                below = background & (np.arange(height)[:, None] >= floor_y)
                pixels[below] = [231, 236, 242]
            picture = caption(pixels, title)
            if not perspective:
                ImageDraw.Draw(picture).line([(0, floor_y), (width, floor_y)],
                                            fill=(137, 151, 168), width=2)
            return picture

        standing_title = (f"Standing | J1 {params['standing_j1_deg']:g} deg, J2 0 deg, "
                          f"J3 {params['standing_j3_deg']:g} deg | +X forward ->")
        frame(standing, standing_title).save(
            output / 'standing_side.png')
        camera.azimuth, camera.elevation, camera.distance = 135, -25, 1.65
        frame(standing, 'Galileo X | outboard shanks and J4 wheel assemblies', True).save(
            output / 'standing_3d.png')
        spin_frames = []
        commands = [4, -3, 2, -1]
        for t in np.arange(60) / 20:
            qpos = standing.copy()
            for leg, speed in zip(params['_legs'], commands):
                qpos[model.joint(f"{leg['name']}_j4").qposadr[0]] = speed * t
            spin_frames.append(frame(qpos, 'Independent spin preview | FL +4, FR -3, RL +2, RR -1 rad/s',
                                     True).resize((900, 600)))
        spin_frames[0].save(output / 'wheel_spin_preview.gif', save_all=True,
                            append_images=spin_frames[1:], duration=50, loop=0)

        camera.azimuth, camera.elevation, camera.distance = 90, 0, 1.4
        sitting_frames = []
        for t in np.linspace(0, 1, 80):
            index = int(np.argmin(np.abs(result.fractions - t)))
            image = frame(result.qpos[index], 'Geometric sitting path | late wheel lift | +X forward ->')
            sitting_frames.append(image.resize((900, 600)))
        frame(result.qpos[-1], f"Torso on floor | J1 {params['sitting_j1_deg']:g} deg, "
              f"J3 {params['sitting_j3_deg']:g} deg | geometry only").save(
            output / 'sitting_side.png')
        sitting_frames[0].save(output / 'sitting_path.gif', save_all=True,
                              append_images=sitting_frames[1:] + [sitting_frames[-1]] * 15,
                              duration=50, loop=0)
        snapshots = [frame(result.qpos[int(np.argmin(abs(result.fractions - t)))],
                           f'Geometric sitting path | {t:.0%}') for t in (0, 0.5, 1)]
        sheet = Image.new('RGB', (1200, 800), 'white')
        for i, image in enumerate(snapshots):
            sheet.paste(image.resize((600, 400)), ((i % 2) * 600, (i // 2) * 400))
        draw = ImageDraw.Draw(sheet)
        draw.text((625, 440), 'Geometry check only', font=font, fill=(35, 49, 70))
        lift = min(result.report['end_wheel_floor_gaps_m'].values()) * 1000
        gap = result.report['minimum']['hardware_floor']['distance_m'] * 1000
        draw.text((625, 480), f'Final wheel lift: {lift:.1f} mm', font=font, fill=(35, 49, 70))
        draw.text((625, 520), f'Min. hardware / floor: {gap:.2f} mm', font=font, fill=(35, 49, 70))
        sheet.save(output / 'sitting_sequence.png')

    data.qpos[:] = standing
    mujoco.mj_forward(model, data)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')
    fig.subplots_adjust(left=0.02, right=0.87, bottom=0.06, top=0.91)
    centre = data.xpos[model.body('torso_link').id]
    corners = np.array([[x, y, z] for x in (-params['torso_lx']/2, params['torso_lx']/2)
                        for y in (-params['torso_ly']/2, params['torso_ly']/2)
                        for z in (-params['torso_lz']/2, params['torso_lz']/2)]) + centre
    for i, a in enumerate(corners):
        for b in corners[i + 1:]:
            if np.sum(np.abs(a-b) > 1e-8) == 1:
                ax.plot(*np.array([a, b]).T, color='#9ba9b8', linewidth=1)
    for leg in params['_legs']:
        name = leg['name']
        anchors = [data.xanchor[model.joint(f'{name}_j{i}').id] for i in (1, 2, 3)]
        shank = data.xpos[model.body(f'{name}_shank_link').id]
        rotation = data.xmat[model.body(f'{name}_shank_link').id].reshape(3, 3)
        tip = shank + rotation @ [0, 0, -params['lower_leg_link_length']]
        wc = data.xanchor[model.joint(f'{name}_j4').id]
        ax.plot(*np.array(anchors + [shank, tip, wc]).T, color='#4b75a8', marker='o', markersize=3)
        theta = np.linspace(0, 2*np.pi, 90)
        points = wc[:, None] + params['wheel_radius'] * (
            rotation[:, 0, None] * np.cos(theta) + rotation[:, 2, None] * np.sin(theta))
        ax.plot(*points, color='#283342', linewidth=1.5)
    for i, color in zip((1, 2, 3, 4), ('#e04b4b', '#32a36b', '#3765cb', '#b348c4')):
        jid = model.joint(f'fl_j{i}').id
        origin, axis = data.xanchor[jid], data.xaxis[jid]
        ax.quiver(*origin, *(axis * 0.12), color=color, arrow_length_ratio=0.25, linewidth=2)
        ax.text(*(origin + axis * 0.135), f'J{i}', color=color, fontsize=11)
    ax.set(xlabel='X / forward (m)', ylabel='Y / left (m)', zlabel='Z / up (m)',
           title='Standing joint axes | J4 stays parallel to moving J3 pitch', zlim=(0, 0.65))
    ax.set_box_aspect((1, 0.8, 0.75))
    ax.view_init(elev=23, azim=-55)
    fig.savefig(output / 'standing_axes.png', dpi=160)
    plt.close(fig)

    profiles = [(cls, params.get(f'{cls}_motor_specification')) for cls in ('J1','J2','J3','J4')]
    profiles = [(cls, spec) for cls, spec in profiles if spec]
    if profiles:
        fig, axes = plt.subplots(3, len(profiles), figsize=(5*len(profiles), 9), squeeze=False)
        for col, (cls, spec) in enumerate(profiles):
            curve = spec['performance_curve']
            for row, (key, label, color) in enumerate([
                    ('speed_rpm','Speed (rpm)','#2b6599'),
                    ('ibus_a','DC bus current (A)','#2b8758'),
                    ('efficiency_pct','Efficiency (%)','#bc603a')]):
                axis = axes[row,col]
                axis.plot(curve['torque_nm'], curve[key], marker='o', color=color, linewidth=1.8)
                axis.axvline(spec['continuous_torque'], color='#9aa7b4', linestyle='--', linewidth=1)
                axis.set(xlabel='Output torque (N m)', ylabel=label, xlim=(0,spec['peak_torque_10s']))
                axis.grid(alpha=.2)
                if row==0: axis.set_title(f"{cls} / {spec['name']}")
        fig.suptitle('User-digitized supplier performance curves (approximate)', fontsize=15)
        fig.tight_layout(rect=(0,0.035,1,.96))
        fig.text(.5,.012,'Dashed: continuous torque. 10 s peak at right endpoint. No extrapolation.',ha='center')
        fig.savefig(output/'motor_curves.png',dpi=160)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    folder = args.model_dir.resolve()
    output = args.output_dir or folder / 'inspection'
    output.mkdir(parents=True, exist_ok=True)
    params_file, xml_file = folder / 'parameters_resolved.yaml', folder / 'galileo_x.xml'
    params = yaml.safe_load(params_file.read_text(encoding='utf-8'))
    model = mujoco.MjModel.from_xml_path(str(xml_file))
    try:
        result = inspect_sitting(model, params, raise_on_error=False)
    except ValueError as exc:
        (output / 'sitting_clearance.json').write_text(
            json.dumps({'valid': False, 'errors': [str(exc)]}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'ERROR {exc}')
        return 2
    report = result.report
    report['model_sha256'] = hashlib.sha256(xml_file.read_bytes()).hexdigest()
    report['parameters_sha256'] = hashlib.sha256(params_file.read_bytes()).hexdigest()
    report['inventory'] = {
        'hinges': sum(int(t == mujoco.mjtJoint.mjJNT_HINGE) for t in model.jnt_type),
        'motor_envelopes': sum(model.geom(i).name.endswith('_motor') for i in range(model.ngeom)),
        'drive_names': [model.actuator(i).name for i in range(model.nu)],
        'wheel_radius_m': float(params['wheel_radius']), 'wheel_width_m': float(params['wheel_width']),
    }
    (output / 'sitting_clearance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'sitting_path.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['fraction'] + [f'qpos_{i}' for i in range(model.nq)])
        writer.writerows([float(t), *q.tolist()] for t, q in zip(result.fractions, result.qpos))
    if not report['valid']:
        print('ERROR ' + '; '.join(report['errors']))
        return 2
    if args.render:
        render_artifacts(folder, output, result, params)
    print(f"PASS {report['samples']} poses; parts {report['minimum']['parts']['distance_m']*1000:.3f} mm; "
          f"hardware/floor {report['minimum']['hardware_floor']['distance_m']*1000:.3f} mm -> {output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
