extends Node
## Headless CI drop test attached to a scene (loads after GDExtension registration).
## Run: godot --headless --path godot/example res://ci_drop_test.tscn

const HALF := 0.25
const DT := 1.0 / 60.0
const STEPS := 180
const Y_TOL := 0.08
const EXPECTED_HASH_HEX := "a0d625f02641a420"

func _ready() -> void:
	if not ClassDB.class_exists("PePhysicsWorld"):
		push_error("ci_drop_test: PePhysicsWorld class missing — GDExtension did not load")
		get_tree().quit(2)
		return
	var world: Node = ClassDB.instantiate("PePhysicsWorld")
	add_child(world)
	world.set("fixed_dt", 0.0)
	world.call("add_ground_plane", 0.0, 0.0, 1.0)
	var id: int = int(world.call("add_box_body", 1.0, HALF, HALF, HALF, 0.0, 2.0, 0.0, 0.0, 0.5))
	if id < 0:
		push_error("ci_drop_test: add_box_body failed")
		get_tree().quit(2)
		return
	for _i in STEPS:
		world.call("step_once", DT)
	var p: Vector3 = world.call("get_body_position", id)
	var h: String = str(world.call("state_hash_hex"))
	print("GODOT_CI_DROP y=", p.y, " hash=", h)
	if abs(p.y - HALF) > Y_TOL:
		push_error("ci_drop_test: final y out of tolerance: %s (want %s ± %s)" % [p.y, HALF, Y_TOL])
		get_tree().quit(1)
		return
	if h != EXPECTED_HASH_HEX:
		push_error("ci_drop_test: state hash mismatch: got %s expected %s" % [h, EXPECTED_HASH_HEX])
		get_tree().quit(1)
		return
	print("GODOT_CI_OK")
	get_tree().quit(0)
