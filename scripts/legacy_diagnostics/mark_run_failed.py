from betting_app.core.db import connect

def mark_run_failed(run_id):
    with connect() as connection:
        connection.execute(
            "UPDATE rating_runs SET status = 'failed', finished_at = CURRENT_TIMESTAMP WHERE id = ?",
            (run_id,),
        )
        print(f"Marked run {run_id} as failed.")

if __name__ == "__main__":
    mark_run_failed(7)
