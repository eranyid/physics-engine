extends Node
## Headless shape authoring smoke: sphere, box, capsule, convex rest near ground.

const DT := 1.0 / 60.0
const STEPS := 240
const Y_TOL := 0.12

func _ready() -> void:
	if not ClassDB.class_exists("PePhysicsWorld"):
		push_error("ci_shapes_test: PePhysicsWorld missing")
		get_tree().quit(2)
		return
	var world: Node = ClassDB.instantiate("PePhysicsWorld")
	add_child(world)
	world.set("fixed_dt", 0.0)
	world.set("interpolate", false)
	world.call("add_ground_plane", 0.0, 0.0, 1.0)

	var sphere_id: int = int(world.call("add_sphere_body", 1.0, 0.25, 0.0, 2.0, 0.0, 0.0, 0.5))
	var box_id: int = int(world.call("add_box_body", 1.0, 0.25, 0.25, 0.25, 1.0, 2.0, 0.0, 0.0, 0.5))
	var cap_id: int = int(world.call("add_capsule_body", 1.0, 0.2, 0.3, 2.0, 2.5, 0.0, 0.0, 0.5))
	var pts := PackedVector3Array([
		Vector3(-0.25, -0.25, -0.25), Vector3(0.25, -0.25, -0.25),
		Vector3(-0.25, 0.25, -0.25), Vector3(0.25, 0.25, -0.25),
		Vector3(-0.25, -0.25, 0.25), Vector3(0.25, -0.25, 0.25),
		Vector3(-0.25, 0.25, 0.25), Vector3(0.25, 0.25, 0.25),
	])
	var convex_id: int = int(world.call("add_convex_body", 1.0, pts, 3.0, 2.0, 0.0, 0.0, 0.5))
	if sphere_id < 0 or box_id < 0 or cap_id < 0 or convex_id < 0:
		push_error("ci_shapes_test: body create failed")
		get_tree().quit(2)
		return

	for _i in STEPS:
		world.call("step_once", DT)

	var ys := [
		world.call("get_body_position", sphere_id).y,
		world.call("get_body_position", box_id).y,
		world.call("get_body_position", cap_id).y,
		world.call("get_body_position", convex_id).y,
	]
	# Sphere r=0.25 → y≈0.25; box half=0.25 → y≈0.25; capsule r+hh along Z so rests on r≈0.2;
	# convex half≈0.25 → y≈0.25.
	var expected := [0.25, 0.25, 0.20, 0.25]
	print("GODOT_CI_SHAPES y=", ys)
	for i in ys.size():
		if abs(ys[i] - expected[i]) > Y_TOL:
			push_error("ci_shapes_test: body %s y=%s want ~%s ± %s" % [i, ys[i], expected[i], Y_TOL])
			get_tree().quit(1)
			return
	print("GODOT_CI_SHAPES_OK")
	get_tree().quit(0)
