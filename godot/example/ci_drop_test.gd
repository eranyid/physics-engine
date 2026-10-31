extends SceneTree
## Headless CI drop test: box on ground, assert final y and state hash.
## Run: godot --headless --path godot/example -s res://ci_drop_test.gd

const HALF := 0.25
const DT := 1.0 / 60.0
const STEPS := 180
const Y_TOL := 0.08
# Locked Rust-side hash from pe_scene::godot_drop_box_result (see docs/GODOT.md).
const EXPECTED_HASH_HEX := "a0d625f02641a420"

func _initialize() -> void:
	var world := PePhysicsWorld.new()
	root.add_child(world)
	world.fixed_dt = 0.0  # step_once only; no physics_process accumulation
	world.add_ground_plane(0.0, 0.0, 1.0)
	var id: int = world.add_box_body(1.0, HALF, HALF, HALF, 0.0, 2.0, 0.0, 0.0, 0.5)
	if id < 0:
		push_error("ci_drop_test: add_box_body failed")
		quit(2)
		return
	for _i in STEPS:
		world.step_once(DT)
	var p: Vector3 = world.get_body_position(id)
	var h: String = str(world.state_hash_hex())
	print("GODOT_CI_DROP y=", p.y, " hash=", h)
	if abs(p.y - HALF) > Y_TOL:
		push_error("ci_drop_test: final y out of tolerance: %s (want %s ± %s)" % [p.y, HALF, Y_TOL])
		quit(1)
		return
	if h != EXPECTED_HASH_HEX:
		push_error("ci_drop_test: state hash mismatch: got %s expected %s" % [h, EXPECTED_HASH_HEX])
		quit(1)
		return
	print("GODOT_CI_OK")
	quit(0)
